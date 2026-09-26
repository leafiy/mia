#!/usr/bin/env python3
"""GPU-0 Laya multilingual sentiment training (supervised CE, not RLCD).

Start on nvidia-smi GPU 0 by UUID, without stopping other services:
  ./start-laya-training-gpu0.sh
Lower memory use while keeping the retained effective batch:
  ./start-laya-training-gpu0.sh --batch-size 8 --grad-accum 32
Defaults: retained best cap100-natural data, 3 epochs, effective batch 256,
mixed precision, gradient checkpointing, seed 42, and pinned Laya base revision.
CUDA/driver access is required; there is no implicit CPU fallback.

Input: laya-sentiment-data-cap100-natural/{train,calibration,test}.jsonl,
verified against manifest. Only train updates weights. Calibration fits one
bounded temperature after the last epoch. Test is evaluated once per run; when
comparing runs, it serves as the development split. Reserve unseen-test.jsonl
for final reporting only. Labels are existing machine predictions, not human
gold labels. --class-weights balanced|sqrt reweights the cross-entropy by
train label counts; the retained runs used none (uniform weights).

Output: laya-sentiment-model-reproduction/final (load with laya.load),
questions.json, metrics.json, history.json, training-config.json and
checkpoint-latest. checkpoint-latest holds the last finished epoch's weights,
NOT optimizer/resume state. --model <local-checkpoint> starts a new fine-tune;
use a new --output. For inference, normalize input with
prepare-laya-sentiment.py:normalize_text and reuse questions.json. Only the
three-way choice head is supervised; act_probability is not a trained
acceptance/rejection signal.

Verification notes: retained cap100-natural and neutral-reviewed-no-uncertain
GPU-0 runs used the pinned HF revision, seed 42, 3 epochs, batch 16, gradient
accumulation 16. A CPU smoke run uses this Python file directly with --device
cpu --limit 5 --epochs 1 --batch-size 2 --grad-accum 2 --output
<new-empty-directory>.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

ROOT = Path(__file__).resolve().parent
MODEL = 'convaiinnovations/laya-multilingual'
REVISION = 'e4e9ddf21a7b1903b7acffd8814ad4307bf63a67'
LABELS = ['负面', '中性', '正面']
INPUT_KEYS = ('input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'laya-sentiment-data-cap100-natural')
    parser.add_argument('--output', type=Path, default=ROOT / 'laya-sentiment-model-reproduction')
    parser.add_argument('--model', default=MODEL, help='Hub model or local Laya checkpoint directory')
    parser.add_argument('--revision', default=REVISION, help='Pinned Hugging Face revision; ignored for a local model')
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--grad-accum', type=int, default=16, help='Default effective batch size: 16 x 16 = 256')
    parser.add_argument('--encoder-lr', type=float, default=2e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='Smoke check only: first N rows of each split; 0 means all')
    parser.add_argument('--class-weights', choices=('none', 'balanced', 'sqrt'), default='none',
                        help='Cross-entropy class weights from train label counts: balanced = N / (K * n_c), '
                             'sqrt = its square root; both rescaled so the count-weighted mean weight is 1')
    args = parser.parse_args()
    for name in ('epochs', 'batch_size', 'grad_accum', 'max_length'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.limit < 0 or args.encoder_lr <= 0 or args.head_lr <= 0:
        parser.error('Limit must be nonnegative and learning rates positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f'Refusing to overwrite nonempty output: {args.output}; choose --output')
    return args


def write_json(path, data):
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def load_items(path, tokenizer, cfg, expected, limit):
    from laya.common import QTYPES, build_sequence
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
        raise ValueError(f'{path}: SHA256 differs from dataset manifest')
    items = []
    schema = None
    with path.open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            state = json.loads(row['state'])
            questions = json.loads(row['questions'])
            gold = json.loads(row['gold'])
            if set(questions) != {'sentiment'} or set(gold) != {'sentiment'}:
                raise ValueError(f'{path}: expected exactly one sentiment question')
            question = questions['sentiment']
            answer = gold['sentiment']
            if question['type'] != 'choice' or list(question['criteria']) != LABELS:
                raise ValueError(f'{path}: unexpected question type or label order')
            if schema is not None and questions != schema:
                raise ValueError(f'{path}: inconsistent sentiment question')
            schema = questions
            label = LABELS.index(answer['label'])
            target = [answer['probabilities'][key] for key in LABELS]
            if target != [float(i == label) for i in range(len(LABELS))]:
                raise ValueError(f'{path}: expected one-hot labels, not estimated probabilities')
            internal = {'t': 'choice', 'ins': question['instructions'], 'crit': question['criteria']}
            # Check the untruncated sequence rather than silently discarding text.
            ids, markers = build_sequence(tokenizer, state, internal, 1_000_000, cfg['head_max_len'])
            if len(ids) > cfg['max_len'] or len(markers) != len(LABELS):
                raise ValueError(f'{row["id"]}: {len(ids)} tokens; increase --max-length (no silent truncation)')
            items.append({'ids': ids, 'markers': markers, 'qtype': QTYPES['choice'],
                          'target': target, 'label': label})
            if limit and len(items) >= limit:
                break
    if not items or (not limit and len(items) != expected['rows']):
        raise ValueError(f'{path}: unexpected or empty row count')
    print(f'DATA {path.name}: {len(items)} rows; max_tokens={max(len(it["ids"]) for it in items)}', flush=True)
    return items, schema


def collate(items, tokenizer, device):
    from laya.common import collate_items
    batch = collate_items([[item] for item in items], tokenizer.pad_token_id)
    return {key: batch[key].to(device) for key in (*INPUT_KEYS, 'label')}


def amp_context(device, dtype):
    import torch
    return torch.autocast('cuda', dtype=dtype) if device.type == 'cuda' else nullcontext()


def collect_logits(model, items, tokenizer, device, dtype, batch_size):
    import torch
    model.eval()
    outputs, labels = [], []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = collate(items[start:start + batch_size], tokenizer, device)
            with amp_context(device, dtype):
                logits, _ = model(*(batch[key] for key in INPUT_KEYS))
            outputs.append(logits.float().cpu())
            labels.append(batch['label'].cpu())
    return torch.cat(outputs), torch.cat(labels)


def fit_temperature(logits, labels):
    import torch.nn.functional as F
    from laya.common import TEMP_MIN, TEMP_MAX
    # Bounded one-dimensional NLL minimization, within the SDK's supported range.
    lo, hi = TEMP_MIN, TEMP_MAX
    ratio = (math.sqrt(5) - 1) / 2
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    loss = lambda value: F.cross_entropy(logits / value, labels).item()
    left_loss, right_loss = loss(left), loss(right)
    for _ in range(40):
        if left_loss < right_loss:
            hi, right, right_loss = right, left, left_loss
            left = hi - ratio * (hi - lo)
            left_loss = loss(left)
        else:
            lo, left, left_loss = left, right, right_loss
            right = lo + ratio * (hi - lo)
            right_loss = loss(right)
    return min((TEMP_MIN, TEMP_MAX, 1.0, (lo + hi) / 2), key=loss)


def metrics(logits, labels, temperature=1.0):
    import torch
    import torch.nn.functional as F
    from laya.common import ece_score
    probs = (logits / temperature).softmax(-1)
    confidence, predicted = probs.max(-1)
    matrix = torch.bincount(labels * 3 + predicted, minlength=9).reshape(3, 3)
    per_class = {}
    for i, name in enumerate(LABELS):
        tp = matrix[i, i].item()
        support, predicted_count = matrix[i].sum().item(), matrix[:, i].sum().item()
        per_class[name] = {'support': support, 'precision': tp / max(1, predicted_count),
                           'recall': tp / max(1, support), 'f1': 2 * tp / max(1, support + predicted_count)}
    return {'rows': len(labels), 'accuracy': (predicted == labels).float().mean().item(),
            'macro_f1': sum(row['f1'] for row in per_class.values()) / 3,
            'nll': F.cross_entropy(logits / temperature, labels).item(),
            'brier': ((probs - F.one_hot(labels, 3)) ** 2).sum(-1).mean().item(),
            'ece': ece_score(confidence.numpy(), (predicted == labels).numpy()),
            'per_class': per_class, 'confusion_matrix': matrix.tolist(), 'label_order': LABELS}


def export_model(path, model, tokenizer, cfg):
    from safetensors.torch import save_file
    path.mkdir(parents=True, exist_ok=True)
    temp = path / 'model.safetensors.partial'
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(temp))
    os.replace(temp, path / 'model.safetensors')
    model.encoder.config.save_pretrained(path / 'encoder')
    tokenizer.save_pretrained(path / 'tokenizer')
    write_json(path / 'rl_agent_config.json', cfg)


def main():
    args = arguments()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        raise RuntimeError('GPU0 is unavailable to PyTorch. Run on the GPU host/container; CPU fallback is disabled.')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        print('GPU', torch.cuda.get_device_name(0), 'CUDA_VISIBLE_DEVICES=' + os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'), flush=True)
    else:
        print('DEVICE cpu (explicit CPU check, not GPU training)', flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16

    manifest = json.loads((args.data_dir / 'manifest.json').read_text(encoding='utf-8'))
    model_dir = Path(args.model)
    if not model_dir.is_dir():
        print('DOWNLOAD', args.model, args.revision, flush=True)
        model_dir = Path(snapshot_download(args.model, revision=args.revision,
                         allow_patterns=['rl_agent_config.json', 'model.safetensors', 'tokenizer/*', 'encoder/*']))
    _fix_tokenizer_config(str(model_dir))
    cfg = json.loads((model_dir / 'rl_agent_config.json').read_text(encoding='utf-8'))
    cfg['max_len'] = args.max_length
    cfg['temperature'] = [1.0, 1.0, 1.0]
    cfg['temperature_by_options'] = {}
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer')
    datasets, question = {}, None
    for split in ('train', 'calibration', 'test'):
        datasets[split], split_question = load_items(args.data_dir / f'{split}.jsonl', tokenizer, cfg,
                                                   manifest['files'][f'{split}.jsonl'], args.limit)
        if question is not None and question != split_question:
            raise ValueError('Question schemas differ between splits')
        question = split_question

    counts = [sum(item['label'] == index for item in datasets['train']) for index in range(len(LABELS))]
    if args.class_weights == 'none':
        weights = [1.0] * len(LABELS)
    else:
        weights = [len(datasets['train']) / (len(LABELS) * max(1, count)) for count in counts]
        if args.class_weights == 'sqrt':
            weights = [math.sqrt(weight) for weight in weights]
        scale = len(datasets['train']) / sum(weight * count for weight, count in zip(weights, counts))
        weights = [weight * scale for weight in weights]
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    print('CLASS_WEIGHTS', args.class_weights, dict(zip(LABELS, [round(weight, 6) for weight in weights])), flush=True)

    model = build_model(cfg, encoder_dir=str(model_dir / 'encoder'), pretrained=False)
    model.load_state_dict(load_file(str(model_dir / 'model.safetensors')), strict=True)
    model.encoder.config.reference_compile = False
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.head_checkpointing = True
    # This task has no act/escalate supervision; do not pretend to train that head.
    for parameter in model.act_head.parameters():
        parameter.requires_grad_(False)
    model.temperature.fill_(1.0)
    model.to(device)
    encoder = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith('encoder.')]
    head = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('encoder.')]
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': args.encoder_lr},
                                  {'params': head, 'lr': args.head_lr}], weight_decay=0.01)
    train = datasets['train']
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(train) / effective_batch)
    total_steps = args.epochs * steps_per_epoch
    warmup = max(1, int(total_steps * 0.06))
    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({'objective': 'supervised cross-entropy; NOT RLCD', 'effective_batch_size': effective_batch,
                   'class_weight_values': dict(zip(LABELS, weights)), 'train_label_counts': dict(zip(LABELS, counts)),
                   'versions': {name: importlib.metadata.version(name) for name in ('laya', 'torch', 'transformers')},
                   'data_manifest': manifest, 'act_head_supervised': False})
    write_json(args.output / 'training-config.json', config)
    started, completed_steps = time.monotonic(), 0
    print(f'TRAIN epochs={args.epochs} updates={total_steps} effective_batch={effective_batch} smoke_limit={args.limit}', flush=True)
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        epoch_loss = 0.0
        for start in range(0, len(order), effective_batch):
            window = order[start:start + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            window_loss = 0.0
            for offset in range(0, len(window), args.batch_size):
                items = [train[index] for index in window[offset:offset + args.batch_size]]
                batch = collate(items, tokenizer, device)
                with amp_context(device, dtype):
                    logits, _ = model(*(batch[key] for key in INPUT_KEYS))
                    loss = F.cross_entropy(logits.float(), batch['label'], weight=class_weight, reduction='sum') / len(window)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Non-finite training loss')
                scaler.scale(loss).backward()
                window_loss += loss.detach().item()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            completed_steps += 1
            epoch_loss += window_loss * len(window)
            if completed_steps == 1 or completed_steps % 20 == 0 or start + len(window) == len(order):
                print(f'PROGRESS epoch={epoch + 1}/{args.epochs} update={completed_steps}/{total_steps} loss={window_loss:.5f} elapsed={time.monotonic() - started:.1f}s', flush=True)
        history.append({'epoch': epoch + 1, 'train_loss': epoch_loss / len(train)})
        cfg['training'] = {'objective': 'supervised_cross_entropy', 'epochs_completed': epoch + 1,
                           'updates': completed_steps, 'fine_tuned_from_checkpoint': True}
        export_model(args.output / 'checkpoint-latest', model, tokenizer, cfg)
        write_json(args.output / 'history.json', history)
        print('CHECKPOINT', args.output / 'checkpoint-latest', flush=True)

    del optimizer, scaler, scheduler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    calibration_logits, calibration_labels = collect_logits(model, datasets['calibration'], tokenizer, device, dtype, args.batch_size)
    temperature = fit_temperature(calibration_logits, calibration_labels)
    test_logits, test_labels = collect_logits(model, datasets['test'], tokenizer, device, dtype, args.batch_size)
    report = {'smoke_limit': args.limit, 'temperature': temperature,
              'calibration_before': metrics(calibration_logits, calibration_labels),
              'calibration_after': metrics(calibration_logits, calibration_labels, temperature),
              'test_before': metrics(test_logits, test_labels),
              'test_after': metrics(test_logits, test_labels, temperature),
              'label_provenance': 'Existing machine-generated sentiment labels, not human gold labels'}
    cfg['temperature'][0] = temperature
    cfg['temperature_by_options'] = {'choice:3-5': temperature}
    model.temperature[0] = temperature
    cfg['amp_dtype'] = 'bf16' if dtype == torch.bfloat16 else 'fp16'
    cfg['sentiment_labels'] = LABELS
    cfg['sentiment_question'] = question
    export_model(args.output / 'final', model, tokenizer, cfg)
    write_json(args.output / 'questions.json', question)
    write_json(args.output / 'metrics.json', report)
    print('COMPLETE', args.output / 'final', 'test_macro_f1=', report['test_after']['macro_f1'],
          'temperature=', temperature, flush=True)


if __name__ == '__main__':
    main()

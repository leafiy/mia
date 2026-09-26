#!/usr/bin/env python3
"""Fine-tune a Qwen3.5 small model as a plain three-way sentiment classifier (no Laya harness).

  CUDA_VISIBLE_DEVICES=<gpu-uuid> .venv-laya/bin/python train-qwen-sentiment.py \
    --model-dir /root/models/qwen3.5/Qwen3.5-0.8B \
    --data-dir laya-sentiment-data-relabeled \
    --output laya-sentiment-model-relabeled-qwen0.8b

Only the text backbone of the multimodal checkpoint is loaded (Qwen3_5TextForSequenceClassification);
the vision tower and lm_head are dropped. Input is "文本：<normalized text>\n情感倾向：" and the class
logits come from the hidden state of the last token (HF sequence-classification pooling, right padding).
Same splits, same metrics.json / unseen-test-metrics.json / chinese-csv-evaluation.json shapes as
train-laya-sentiment.py so compare / summary scripts work unchanged. Labels are the dataset's gold
labels (for the relabeled dataset: llm labels). Output final/ holds bf16 weights loadable
with Qwen3_5TextForSequenceClassification.from_pretrained plus sentiment-config.json (labels, template,
temperature).
"""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

ROOT = Path(__file__).resolve().parent
LABELS = ['负面', '中性', '正面']
TEMPLATE = '文本：{text}\n情感倾向：'
CSVS = tuple(ROOT.parent / 'data' / 'eval' / f'中文情感测试集_{n:02d}.csv' for n in (1, 2, 3))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir', type=Path, required=True, help='Local Qwen3.5 checkpoint directory')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'laya-sentiment-data-relabeled')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16, help='Micro-batch')
    parser.add_argument('--grad-accum', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5, help='Backbone learning rate')
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class-weights', choices=('none', 'balanced'), default='none')
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--limit', type=int, default=0, help='Smoke check: first N rows of each split')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f'Refusing to overwrite nonempty output: {args.output}')
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def load_split(data_dir, split, manifest, limit):
    path = data_dir / f'{split}.jsonl'
    expected = manifest['files'][f'{split}.jsonl']
    if sha256_file(path) != expected['sha256']:
        raise ValueError(f'{path}: SHA256 differs from manifest')
    texts, labels = [], []
    with path.open(encoding='utf-8') as f:
        for n, line in enumerate(f):
            if limit and n >= limit:
                break
            row = json.loads(line)
            texts.append(json.loads(row['state'])['text'])
            labels.append(LABELS.index(json.loads(row['gold'])['sentiment']['label']))
    if not limit and len(texts) != expected['rows']:
        raise ValueError(f'{path}: row count differs from manifest')
    return texts, labels


def encode(tokenizer, texts, max_length):
    enc = tokenizer([TEMPLATE.format(text=t) for t in texts], add_special_tokens=False)
    ids = enc['input_ids']
    longest = max(len(x) for x in ids)
    if longest > max_length:
        raise ValueError(f'longest sequence {longest} > --max-length {max_length}; no silent truncation')
    return ids


def batches(ids, labels, order, batch_size, pad_id, device):
    import torch
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        width = max(len(ids[i]) for i in idx)
        input_ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):
            input_ids[r, :len(ids[i])] = torch.tensor(ids[i])
            attention[r, :len(ids[i])] = 1
        yield (input_ids.to(device), attention.to(device),
               torch.tensor([labels[i] for i in idx], dtype=torch.long, device=device))


def collect_logits(model, ids, labels, batch_size, pad_id, device):
    import torch
    model.eval()
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = torch.empty((len(ids), len(LABELS)), dtype=torch.float32)
    with torch.inference_mode():
        pos = 0
        for input_ids, attention, _ in batches(ids, labels, order, batch_size, pad_id, device):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attention).logits
            n = input_ids.shape[0]
            out[order[pos:pos + n]] = logits.float().cpu()
            pos += n
    return out, torch.tensor(labels, dtype=torch.long)


def fit_temperature(logits, labels):
    import torch.nn.functional as F
    lo, hi = 0.5, 5.0
    ratio = (math.sqrt(5) - 1) / 2
    loss = lambda t: F.cross_entropy(logits / t, labels).item()
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fl, fr = loss(left), loss(right)
    for _ in range(40):
        if fl < fr:
            hi, right, fr = right, left, fl
            left = hi - ratio * (hi - lo)
            fl = loss(left)
        else:
            lo, left, fl = left, right, fr
            right = lo + ratio * (hi - lo)
            fr = loss(right)
    return min((0.5, 5.0, 1.0, (lo + hi) / 2), key=loss)


def ece_score(confidence, correct, bins=15):
    import numpy as np
    confidence, correct = np.asarray(confidence), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def metrics(logits, labels, temperature=1.0):
    import torch
    import torch.nn.functional as F
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
            'macro_f1': sum(r['f1'] for r in per_class.values()) / 3,
            'nll': F.cross_entropy(logits / temperature, labels).item(),
            'brier': ((probs - F.one_hot(labels, 3)) ** 2).sum(-1).mean().item(),
            'ece': ece_score(confidence.numpy(), (predicted == labels).numpy()),
            'per_class': per_class, 'confusion_matrix': matrix.tolist(), 'label_order': LABELS}


def load_backbone(model_dir, num_labels, pad_id):
    """Text-only sequence classifier initialised from the multimodal checkpoint's language model."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import Qwen3_5TextForSequenceClassification
    full = AutoConfig.from_pretrained(model_dir)
    text_cfg = full.text_config
    text_cfg.num_labels = num_labels
    text_cfg.pad_token_id = pad_id
    text_cfg.use_cache = False
    text_cfg.dtype = torch.float32
    model = Qwen3_5TextForSequenceClassification(text_cfg)
    state = {}
    for shard in sorted(Path(model_dir).glob('*.safetensors')):
        for key, value in load_file(str(shard)).items():
            if key.startswith('model.language_model.'):
                state['model.' + key[len('model.language_model.'):]] = value.float()
            elif key.startswith(('model.visual.', 'lm_head.', 'mtp.')):
                # vision tower, tied LM head and multi-token-prediction head are not used by the classifier
                continue
            else:
                state[key] = value.float()
    result = model.load_state_dict(state, strict=False)
    unexpected = [k for k in result.unexpected_keys]
    missing = [k for k in result.missing_keys if not k.startswith('score.')]
    if unexpected or missing:
        raise RuntimeError(f'weight mapping mismatch: missing={missing[:5]} unexpected={unexpected[:5]}')
    print(f'BACKBONE loaded {len(state)} tensors; head initialised: {result.missing_keys}', flush=True)
    return model


def main():
    args = arguments()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device = torch.device(args.device)
    torch.cuda.set_device(0)
    print('GPU', torch.cuda.get_device_name(0), 'CUDA_VISIBLE_DEVICES=' + os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'), flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = json.loads((args.data_dir / 'manifest.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    pad_id = tokenizer.pad_token_id
    data = {}
    for split in ('train', 'calibration', 'test', 'unseen-test'):
        texts, labels = load_split(args.data_dir, split, manifest, args.limit)
        ids = encode(tokenizer, texts, args.max_length)
        data[split] = (ids, labels, texts)
        print(f'DATA {split}: {len(ids)} rows; max_tokens={max(len(x) for x in ids)}', flush=True)

    train_ids, train_labels, _ = data['train']
    counts = [train_labels.count(i) for i in range(len(LABELS))]
    if args.class_weights == 'balanced':
        weights = [len(train_labels) / (len(LABELS) * max(1, c)) for c in counts]
    else:
        weights = [1.0] * len(LABELS)
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    print('CLASS_WEIGHTS', args.class_weights, dict(zip(LABELS, [round(w, 6) for w in weights])), flush=True)

    model = load_backbone(args.model_dir, len(LABELS), pad_id)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'MODEL params={n_params/1e6:.1f}M', flush=True)
    head = [p for n, p in model.named_parameters() if n.startswith('score.')]
    backbone = [p for n, p in model.named_parameters() if not n.startswith('score.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': args.lr}, {'params': head, 'lr': args.head_lr}], weight_decay=0.01)
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(train_ids) / effective_batch)
    total_steps = args.epochs * steps_per_epoch
    warmup = max(1, int(total_steps * 0.06))

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'objective': 'supervised cross-entropy; Qwen3.5 text backbone + sequence-classification head',
                   'template': TEMPLATE, 'pooling': 'last non-pad token (HF sequence classification)',
                   'effective_batch_size': effective_batch, 'params_million': round(n_params / 1e6, 1),
                   'class_weight_values': dict(zip(LABELS, weights)), 'train_label_counts': dict(zip(LABELS, counts)),
                   'versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers')},
                   'base_model': str(args.model_dir), 'data_manifest': manifest})
    write_json(args.output / 'training-config.json', config)
    print(f'TRAIN epochs={args.epochs} updates={total_steps} effective_batch={effective_batch} smoke_limit={args.limit}', flush=True)

    history, started, step = [], time.monotonic(), 0
    for epoch in range(args.epochs):
        order = list(range(len(train_ids)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        epoch_loss = 0.0
        for start in range(0, len(order), effective_batch):
            window = order[start:start + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            window_loss = 0.0
            for input_ids, attention, labels in batches(train_ids, train_labels, window, args.batch_size, pad_id, device):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(input_ids=input_ids, attention_mask=attention).logits
                loss = F.cross_entropy(logits.float(), labels, weight=class_weight, reduction='sum') / len(window)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                window_loss += loss.detach().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_loss += window_loss * len(window)
            if step == 1 or step % 20 == 0 or start + len(window) == len(order):
                print(f'PROGRESS epoch={epoch + 1}/{args.epochs} update={step}/{total_steps} loss={window_loss:.5f} elapsed={time.monotonic() - started:.1f}s', flush=True)
        dev_logits, dev_labels = collect_logits(model, data['test'][0], data['test'][1], args.eval_batch_size, pad_id, device)
        dev = metrics(dev_logits, dev_labels)
        history.append({'epoch': epoch + 1, 'train_loss': epoch_loss / len(train_ids),
                        'dev_accuracy': dev['accuracy'], 'dev_macro_f1': dev['macro_f1']})
        write_json(args.output / 'history.json', history)
        print(f'EPOCH {epoch + 1} train_loss={history[-1]["train_loss"]:.5f} dev_acc={dev["accuracy"]:.4f} dev_macro_f1={dev["macro_f1"]:.4f}', flush=True)

    del optimizer, scheduler
    torch.cuda.empty_cache()
    cal_logits, cal_labels = collect_logits(model, data['calibration'][0], data['calibration'][1], args.eval_batch_size, pad_id, device)
    temperature = fit_temperature(cal_logits, cal_labels)
    test_logits, test_labels = collect_logits(model, data['test'][0], data['test'][1], args.eval_batch_size, pad_id, device)
    report = {'smoke_limit': args.limit, 'temperature': temperature,
              'calibration_before': metrics(cal_logits, cal_labels), 'calibration_after': metrics(cal_logits, cal_labels, temperature),
              'test_before': metrics(test_logits, test_labels), 'test_after': metrics(test_logits, test_labels, temperature),
              'label_provenance': manifest.get('fields_used_as_text_or_target', {}).get('target', 'dataset gold labels')}
    write_json(args.output / 'metrics.json', report)
    write_json(args.output / 'questions.json', {'sentiment': {'type': 'choice', 'instructions': '判断文本表达的整体情感倾向。',
                                                             'criteria': {'负面': '表达不满、失望、批评等负面态度', '中性': '态度客观或情绪不明显', '正面': '表达满意、赞赏、愉快等正面态度'}}})

    unseen_logits, unseen_labels = collect_logits(model, data['unseen-test'][0], data['unseen-test'][1], args.eval_batch_size, pad_id, device)
    unseen_path = args.data_dir / 'unseen-test.jsonl'
    write_json(args.output / 'unseen-test-metrics.json', {
        'model': str((args.output / 'final').resolve()), 'requested_device': args.device, 'actual_device': str(device),
        'training': {'objective': 'supervised_cross_entropy', 'epochs_completed': args.epochs, 'updates': step},
        'temperature': [temperature, 1.0, 1.0],
        'dataset': {'path': str(unseen_path.resolve()), 'sha256': manifest['files']['unseen-test.jsonl']['sha256'],
                    'manifest_rows': manifest['files']['unseen-test.jsonl']['rows'], 'evaluated_rows': len(unseen_labels), 'limit': args.limit},
        'metrics': metrics(unseen_logits, unseen_labels, temperature),
        'label_provenance': report['label_provenance']})

    spec = importlib.util.spec_from_file_location('prep', ROOT / 'prepare-laya-sentiment.py')
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    train_texts = set(data['train'][2])
    csv_report = {'model': str((args.output / 'final').resolve()), 'actual_device': str(device),
                  'training_data': str((args.data_dir / 'train.jsonl').resolve()), 'labels': LABELS, 'datasets': {}}
    for path in CSVS:
        rows = []
        with path.open(newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                rows.append((prep.normalize_text(row['text']), row['label'].strip()))
        ids = encode(tokenizer, [t for t, _ in rows], args.max_length)
        labels = [LABELS.index(l) for _, l in rows]
        logits, gold = collect_logits(model, ids, labels, args.eval_batch_size, pad_id, device)
        csv_report['datasets'][path.name] = {'path': str(path.resolve()), 'label_counts': dict(Counter(l for _, l in rows)),
                                             'exact_train_text_overlap': sum(t in train_texts for t, _ in rows),
                                             'metrics': metrics(logits, gold, temperature)}
        m = csv_report['datasets'][path.name]['metrics']
        print(f'CSV {path.name} acc={m["accuracy"]:.4f} macro_f1={m["macro_f1"]:.4f} neutral_recall={m["per_class"]["中性"]["recall"]:.3f}', flush=True)
    write_json(args.output / 'chinese-csv-evaluation.json', csv_report)

    final = args.output / 'final'
    model.to(torch.bfloat16).save_pretrained(final, safe_serialization=True)
    tokenizer.save_pretrained(final)
    write_json(final / 'sentiment-config.json', {'labels': LABELS, 'template': TEMPLATE, 'temperature': temperature,
                                                 'pooling': 'last non-pad token', 'base_model': str(args.model_dir),
                                                 'load_with': 'transformers.models.qwen3_5.Qwen3_5TextForSequenceClassification'})
    print('COMPLETE', final, 'test_macro_f1=', report['test_after']['macro_f1'], 'temperature=', temperature, flush=True)


if __name__ == '__main__':
    main()

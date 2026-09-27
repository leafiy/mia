#!/usr/bin/env python3
"""Evaluate a CLM head (CLM's finetune.py output, or the reference head zero-shot) on Mia's evaluation sets,
writing the same report files as the other Mia versions.

  .venv-laya/bin/python scripts/evaluate-clm-sentiment.py --clm-repo /path/to/CLM --embed-model /path/to/Qwen3-8B \
    --head <run>/best_head.pt --cache <workdir>/embeddings/eval.npz --train-data <train.jsonl> --out-dir reports/<alias>
  ... --head CLM_v0.1-8B.pt --no-calibration --out-dir reports/clm-v0.1-8b-zero-shot
  ... --head <run>/best_head.pt --throughput --out-dir reports/<alias>

Each row becomes the typed question the model was trained on (state {"text": ...} plus the dataset's questions
JSON), so the state head sees "text: <文本>\n\n判断文本表达的整体情感倾向。" and the action head the three criteria
descriptions. Logits are exp(logit_scale) * cos. Unless --no-calibration, one temperature is fitted on
calibration.jsonl (the Qwen trainer's golden-section search) and applied to every other set; it is stored in
<out-dir>/sentiment-config.json, which evaluate-public-benchmarks.py --model-type clm reads.
--throughput times the whole path on the holdout texts without the cache: tokenise, Qwen3-8B forward, heads,
softmax (the three option vectors are embedded once beforehand, as CLM serves them).
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parent
EVAL = ROOT.parent / 'data' / 'eval'
LABELS = ['负面', '中性', '正面']
SPLITS = (('calibration', 'calibration.jsonl'), ('dev', 'test.jsonl'), ('holdout', 'unseen-test.jsonl'))


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_split(path, manifest):
    expected = manifest['files'][path.name]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected['sha256']:
        raise ValueError(f'{path}: SHA-256 differs from manifest')
    texts, labels, questions = [], [], None
    for line in path.open(encoding='utf-8'):
        row = json.loads(line)
        texts.append(json.loads(row['state'])['text'])
        labels.append(LABELS.index(json.loads(row['gold'])['sentiment']['label']))
        questions = questions or json.loads(row['questions'])
    if len(texts) != expected['rows']:
        raise ValueError(f'{path}: row count differs from manifest')
    return {'path': path, 'sha256': digest, 'rows': expected['rows'], 'texts': texts, 'labels': labels,
            'questions': questions, 'provenance': manifest.get('fields_used_as_text_or_target', {}).get('target', 'dataset gold labels')}


class ClmSentiment:
    """Chinese text -> 负面/中性/正面 logits through a CLM head over cached Qwen3-8B embeddings."""

    def __init__(self, clm_repo, embed_model, head, device='cuda:0', batch_size=64, cache=None, questions=None):
        embed = module('clm-embed.py', 'clm_embed')
        self.embed, self.clm = embed, embed.clm_import(clm_repo)
        self.schema, heads = self.clm[3], self.clm[4]
        self.questions = questions or json.loads(next((EVAL / 'test.jsonl').open(encoding='utf-8')))['questions']
        if isinstance(self.questions, str):
            self.questions = json.loads(self.questions)
        _, keys, self.options = self.schema.build_pairs({'text': ''}, self.questions)['sentiment']
        if keys != LABELS:
            raise ValueError(f'option order {keys} differs from {LABELS}')
        self.pair = heads.HeadPair('head', str(head), device=device).ensure()
        self.encoder = embed.CachedEncoder(self.clm, embed_model, cache, device, batch_size) if cache else None

    def state(self, text):
        return self.schema.build_pairs({'text': text}, self.questions)['sentiment'][0]

    def ensure(self, texts):
        self.encoder.ensure(list(dict.fromkeys([self.state(t) for t in texts] + list(self.options))))

    def logits(self, texts):
        import torch
        self.ensure(texts)
        states = self.encoder.vectors([self.state(t) for t in texts])
        return torch.tensor(self.embed.choice_logits(self.pair, states, self.encoder.vectors(self.options)), dtype=torch.float32)


def throughput(args, holdout):
    import torch
    scorer = ClmSentiment(args.clm_repo, args.embed_model, args.head, args.device, args.batch_size)
    recipe = scorer.clm[2].Recipe(str(args.embed_model), 2048)
    embedder = scorer.embed.LastTokenEmbedder(args.embed_model, args.device, args.batch_size)
    zc = scorer.pair.project_actions(embedder.embed([recipe.text_ids(t, keep='tail') for t in scorer.options]))

    def run(texts):
        ids = [recipe.text_ids(scorer.state(t), keep='tail') for t in texts]
        zs = scorer.pair.project_states(embedder.embed(ids))
        return (scorer.pair.scale * zs @ zc.T).softmax(-1).cpu()

    run(holdout['texts'][:512])
    torch.cuda.synchronize()
    started = time.time()
    probs = run(holdout['texts'])
    torch.cuda.synchronize()
    seconds = time.time() - started
    acc = (probs.argmax(-1) == torch.tensor(holdout['labels'])).float().mean().item()
    report = {'head': str(args.head), 'embed_model': str(args.embed_model), 'device': torch.cuda.get_device_name(args.device),
              'dtype': 'bfloat16', 'batch_size': args.batch_size, 'rows': len(holdout['texts']), 'seconds': seconds,
              'rows_per_second': len(holdout['texts']) / seconds, 'holdout_accuracy_check': acc,
              'path': 'tokenise + Qwen3-8B forward (transformers, sdpa) + both heads + softmax; option vectors precomputed'}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--clm-repo', type=Path, required=True)
    parser.add_argument('--embed-model', type=Path, required=True, help='Qwen3-8B checkpoint directory')
    parser.add_argument('--head', type=Path, required=True, help='CLM head checkpoint (best_head.pt or CLM_v0.1-8B.pt)')
    parser.add_argument('--cache', type=Path, help='TextCache .npz for evaluation embeddings (shared across heads)')
    parser.add_argument('--train-data', type=Path, help='train.jsonl for the exact-text overlap audit of the CSVs')
    parser.add_argument('--no-calibration', action='store_true', help='keep temperature 1 (zero-shot reference head)')
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.throughput and not args.cache:
        parser.error('--cache is required unless --throughput')

    trainer = module('train-qwen-sentiment.py', 'qwen_sentiment_trainer')
    manifest = json.loads((EVAL / 'manifest.json').read_text(encoding='utf-8'))
    splits = {name: load_split(EVAL / file, manifest) for name, file in SPLITS}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.throughput:
        trainer.write_json(args.out_dir / 'throughput.json', throughput(args, splits['holdout']))
        return

    import torch
    prep = module('prepare-laya-sentiment.py', 'laya_sentiment_preparation')
    scorer = ClmSentiment(args.clm_repo, args.embed_model, args.head, args.device, args.batch_size, args.cache,
                          splits['dev']['questions'])
    csvs = []
    for path in trainer.CSVS:
        with path.open(newline='', encoding='utf-8-sig') as f:
            csvs.append((path, [(prep.normalize_text(r['text']), r['label'].strip()) for r in csv.DictReader(f)]))
    scorer.ensure([t for s in splits.values() for t in s['texts']] + [t for _, rows in csvs for t, _ in rows])

    cal = splits['calibration']
    cal_logits, cal_labels = scorer.logits(cal['texts']), torch.tensor(cal['labels'])
    temperature = 1.0 if args.no_calibration else trainer.fit_temperature(cal_logits, cal_labels)
    config = {'labels': LABELS, 'questions': scorer.questions, 'temperature': temperature,
              'head': str(args.head.resolve()), 'logit_scale_exp': scorer.pair.scale,
              'embed_model': str(args.embed_model), 'pooling': 'Qwen3-8B last token, L2-normalised',
              'state_text_example': scorer.state('<文本>'), 'options': list(scorer.options),
              'calibrated_on': None if args.no_calibration else str(cal['path'])}
    trainer.write_json(args.out_dir / 'sentiment-config.json', config)
    trainer.write_json(args.out_dir / 'calibration-metrics.json', {
        'temperature': temperature, 'before': trainer.metrics(cal_logits, cal_labels),
        'after': trainer.metrics(cal_logits, cal_labels, temperature)})

    for name, filename in (('dev', 'eval-dev-metrics.json'), ('holdout', 'eval-holdout-metrics.json')):
        split = splits[name]
        m = trainer.metrics(scorer.logits(split['texts']), torch.tensor(split['labels']), temperature)
        trainer.write_json(args.out_dir / filename, {
            'model': str(args.head.resolve()), 'requested_device': args.device, 'actual_device': args.device,
            'temperature': [temperature, 1.0, 1.0],
            'dataset': {'path': str(split['path'].resolve()), 'sha256': split['sha256'], 'manifest_rows': split['rows'],
                        'evaluated_rows': len(split['texts']), 'limit': 0},
            'metrics': m, 'label_provenance': split['provenance']})
        print(f"{name}: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} neutral_recall={m['per_class']['中性']['recall']:.3f} ece={m['ece']:.4f}", flush=True)

    train_texts = set()
    if args.train_data:
        for line in args.train_data.open(encoding='utf-8'):
            train_texts.add(json.loads(json.loads(line)['state'])['text'])
    report = {'model': str(args.head.resolve()), 'actual_device': args.device,
              'training_data': str(args.train_data.resolve()) if args.train_data else None, 'labels': LABELS, 'datasets': {}}
    for path, rows in csvs:
        m = trainer.metrics(scorer.logits([t for t, _ in rows]), torch.tensor([LABELS.index(l) for _, l in rows]), temperature)
        report['datasets'][path.name] = {'path': str(path.resolve()), 'label_counts': dict(Counter(l for _, l in rows)),
                                         'exact_train_text_overlap': sum(t in train_texts for t, _ in rows) if args.train_data else None,
                                         'metrics': m}
        print(f"{path.name}: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} neutral_recall={m['per_class']['中性']['recall']:.3f}", flush=True)
    trainer.write_json(args.out_dir / 'chinese-csv-evaluation.json', report)
    print(f'temperature {temperature:.4f}; reports in {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()

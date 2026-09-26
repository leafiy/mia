#!/usr/bin/env python3
"""Evaluate a finished Qwen3.5 sentiment classifier (train-qwen-sentiment.py output) on a manifest-listed
JSONL split and/or the three Chinese CSVs, writing the same report shapes as test-laya-sentiment.py and
evaluate-laya-chinese-csv.py.

  CUDA_VISIBLE_DEVICES=<gpu-uuid> .venv-laya/bin/python evaluate-qwen-sentiment.py \
    --model laya-sentiment-model-relabeled-qwen3.5-2b/final \
    --data laya-sentiment-data-relabeled-mixed-neutral/test.jsonl \
    --output laya-sentiment-model-relabeled-qwen3.5-2b/relabeled-mixed-neutral-test-metrics.json

  ... --csv --train-data <train.jsonl for the overlap audit> --output <chinese-csv-evaluation.json>

Reports are always printed; --output must not point at an existing report you want to keep.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parent


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True, help='final/ directory written by train-qwen-sentiment.py')
    parser.add_argument('--data', type=Path, help='Manifest-listed JSONL split')
    parser.add_argument('--manifest', type=Path, help='Dataset manifest (default: manifest.json beside --data)')
    parser.add_argument('--csv', action='store_true', help='Evaluate the three 中文情感测试集 CSVs instead')
    parser.add_argument('--train-data', type=Path, help='train.jsonl for the exact-text overlap audit (--csv)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if bool(args.data) == bool(args.csv):
        parser.error('give exactly one of --data or --csv')
    if args.data and not args.manifest:
        args.manifest = args.data.parent / 'manifest.json'
    return args


def main():
    args = arguments()
    import torch
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5TextForSequenceClassification
    trainer = module('train-qwen-sentiment.py', 'qwen_sentiment_trainer')
    cfg = json.loads((args.model / 'sentiment-config.json').read_text(encoding='utf-8'))
    if cfg['labels'] != trainer.LABELS or cfg['template'] != trainer.TEMPLATE:
        raise ValueError('sentiment-config.json labels/template differ from train-qwen-sentiment.py')
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5TextForSequenceClassification.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    temperature = float(cfg['temperature'])
    pad_id = tokenizer.pad_token_id

    def run(texts, labels):
        ids = trainer.encode(tokenizer, texts, 4096)
        logits, gold = trainer.collect_logits(model, ids, labels, args.batch_size, pad_id, device)
        return trainer.metrics(logits, gold, temperature)

    if args.data:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        expected = manifest['files'][args.data.name]
        digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
        if digest != expected['sha256']:
            raise ValueError(f'{args.data}: SHA-256 differs from manifest')
        texts, labels = [], []
        with args.data.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                texts.append(json.loads(row['state'])['text'])
                labels.append(trainer.LABELS.index(json.loads(row['gold'])['sentiment']['label']))
        if len(texts) != expected['rows']:
            raise ValueError(f'{args.data}: row count differs from manifest')
        report = {'model': str(args.model.resolve()), 'requested_device': args.device, 'actual_device': str(device),
                  'temperature': [temperature, 1.0, 1.0],
                  'dataset': {'path': str(args.data.resolve()), 'sha256': digest, 'manifest_rows': expected['rows'],
                              'evaluated_rows': len(texts), 'limit': 0},
                  'metrics': run(texts, labels),
                  'label_provenance': manifest.get('fields_used_as_text_or_target', {}).get('target', 'dataset gold labels')}
    else:
        prep = module('prepare-laya-sentiment.py', 'laya_sentiment_preparation')
        train_texts = set()
        if args.train_data:
            with args.train_data.open(encoding='utf-8') as f:
                for line in f:
                    train_texts.add(json.loads(json.loads(line)['state'])['text'])
        report = {'model': str(args.model.resolve()), 'actual_device': str(device),
                  'training_data': str(args.train_data.resolve()) if args.train_data else None,
                  'labels': trainer.LABELS, 'datasets': {}}
        for path in trainer.CSVS:
            rows = []
            with path.open(newline='', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    rows.append((prep.normalize_text(row['text']), row['label'].strip()))
            report['datasets'][path.name] = {
                'path': str(path.resolve()), 'label_counts': dict(Counter(l for _, l in rows)),
                'exact_train_text_overlap': sum(t in train_texts for t, _ in rows),
                'metrics': run([t for t, _ in rows], [trainer.LABELS.index(l) for _, l in rows])}
    if args.output:
        trainer.write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

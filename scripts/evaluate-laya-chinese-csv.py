#!/usr/bin/env python3
"""Evaluate the same trained Laya checkpoint against three labeled CSV datasets.

CUDA_VISIBLE_DEVICES=<free-GPU-UUID> .venv-laya/bin/python \
  evaluate-laya-chinese-csv.py --device cuda:0 --batch-size 64

Each CSV needs id, text, and label columns. Input text is normalized exactly as in
training. Reported scores compare against the CSV labels, not machine train labels.
"""
import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / 'models' / 'mia-laya' / 'final'
TRAIN_DATA = None  # training set is not distributed; pass --train-data for the overlap audit
DEFAULT_CSVS = tuple(ROOT.parent / 'data' / 'eval' / f'中文情感测试集_{n:02d}.csv' for n in (1, 2, 3))


def script_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csvs', nargs='*', type=Path, default=DEFAULT_CSVS)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--train-data', type=Path, default=TRAIN_DATA,
                        help='Training JSONL used for exact normalized-text overlap audit')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', type=Path,
                        help='Optional JSON report path; report is always printed to stdout')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    return args


def training_texts(path):
    texts = set()
    with path.open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            texts.add(json.loads(row['state'])['text'])
    return texts


def load_csv(path, normalize_text, label_order):
    rows = []
    seen_ids = set()
    with path.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {'id', 'text', 'label'}.issubset(reader.fieldnames):
            raise ValueError(f'{path}: expected id, text and label columns')
        for line_number, row in enumerate(reader, start=2):
            row_id = row['id']
            if not row_id or row_id in seen_ids:
                raise ValueError(f'{path}:{line_number}: missing or repeated id {row_id!r}')
            seen_ids.add(row_id)
            label = row['label'].strip()
            if label not in label_order:
                raise ValueError(f'{path}:{line_number}: unexpected label {label!r}')
            text = normalize_text(row['text'])
            if not text:
                raise ValueError(f'{path}:{line_number}: empty normalized text')
            rows.append((row_id, text, label, row.get('industry', '')))
    if not rows:
        raise ValueError(f'{path}: no examples')
    return rows


def main():
    args = arguments()
    if not args.model.is_dir():
        raise FileNotFoundError(f'Finished model not found: {args.model}')
    if args.train_data and not args.train_data.is_file():
        raise FileNotFoundError(f'Training data not found: {args.train_data}')

    preparation = script_module('prepare-laya-sentiment.py', 'laya_sentiment_preparation')
    evaluator = script_module('test-laya-sentiment.py', 'laya_sentiment_evaluator')
    questions = json.loads((args.model.parent / 'questions.json').read_text(encoding='utf-8'))
    if tuple(questions['sentiment']['criteria']) != evaluator.LABELS:
        raise ValueError('Checkpoint question label order differs from evaluator')
    train_texts = training_texts(args.train_data) if args.train_data else set()
    from laya import load
    agent = load(str(args.model.resolve()), device=args.device)

    report = {
        'model': str(args.model.resolve()),
        'actual_device': str(agent.device),
        'training_data': str(args.train_data.resolve()) if args.train_data else None,
        'labels': list(evaluator.LABELS),
        'datasets': {},
    }
    for path in args.csvs:
        rows = load_csv(path, preparation.normalize_text, evaluator.LABELS)
        states = [{'text': row[1]} for row in rows]
        gold = [evaluator.LABELS.index(row[2]) for row in rows]
        metrics = evaluator.evaluate(agent, states, gold, questions, args.batch_size)
        report['datasets'][path.name] = {
            'path': str(path.resolve()),
            'label_counts': dict(Counter(row[2] for row in rows)),
            'exact_train_text_overlap': sum(row[1] in train_texts for row in rows),
            'metrics': metrics,
        }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        evaluator.write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

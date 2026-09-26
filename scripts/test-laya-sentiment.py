#!/usr/bin/env python3
"""Evaluate the finished Laya sentiment model on the fixed unseen test set.

CPU (does not compete with GPU 0 training):
  .venv-laya/bin/python test-laya-sentiment.py

After training has finished, a free GPU can be selected explicitly:
  CUDA_VISIBLE_DEVICES=1 .venv-laya/bin/python test-laya-sentiment.py \
    --device cuda:0 --batch-size 64

The default dataset contains every eligible row left in the test-only source-group
pool after train, calibration, and the 5,000-row development set were selected.
Its SHA-256 and row count are checked against laya-sentiment-data-cap100-natural/manifest.json.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

ROOT = Path(__file__).resolve().parent
LABELS = ('负面', '中性', '正面')
DEFAULT_MODEL = ROOT.parent / 'models' / 'mia-laya' / 'final'
DEFAULT_DATA = ROOT.parent / 'data' / 'eval' / 'unseen-test.jsonl'


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL,
                        help='Finished Laya checkpoint (default: models/mia-laya/final)')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA,
                        help='Manifest-listed Laya JSONL test set')
    parser.add_argument('--manifest', type=Path,
                        help='Dataset manifest (default: manifest.json beside --data)')
    parser.add_argument('--device', default='cpu',
                        help='Laya inference device, for example cpu or cuda:0 (default: cpu)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='States per model forward pass (default: 16)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Evaluate only the first N rows for a smoke run; 0 evaluates all')
    parser.add_argument('--output', type=Path,
                        help='Optional JSON report path; the report is always printed to stdout')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    args.manifest = args.manifest or args.data.parent / 'manifest.json'
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path, expected, limit):
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected['sha256']:
        raise ValueError(f'{path}: SHA-256 differs from {expected["sha256"]}')

    states, labels, ids = [], [], set()
    questions = None
    with path.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            row_id = row['id']
            if row_id in ids:
                raise ValueError(f'{path}:{line_number}: duplicate id {row_id!r}')
            ids.add(row_id)
            row_questions = json.loads(row['questions'])
            gold = json.loads(row['gold'])
            if set(row_questions) != {'sentiment'} or set(gold) != {'sentiment'}:
                raise ValueError(f'{path}:{line_number}: expected exactly one sentiment question')
            question = row_questions['sentiment']
            answer = gold['sentiment']
            if question.get('type') != 'choice' or tuple(question.get('criteria', ())) != LABELS:
                raise ValueError(f'{path}:{line_number}: unexpected question schema or label order')
            label = answer.get('label')
            if label not in LABELS:
                raise ValueError(f'{path}:{line_number}: unexpected gold label {label!r}')
            expected_probabilities = {name: float(name == label) for name in LABELS}
            if answer.get('probabilities') != expected_probabilities:
                raise ValueError(f'{path}:{line_number}: gold probabilities are not one-hot')
            if questions is not None and row_questions != questions:
                raise ValueError(f'{path}:{line_number}: question schema differs from earlier rows')
            questions = row_questions
            states.append(json.loads(row['state']))
            labels.append(LABELS.index(label))
            if limit and len(states) >= limit:
                break

    if not states:
        raise ValueError(f'{path}: dataset is empty')
    if not limit and len(states) != expected['rows']:
        raise ValueError(f'{path}: found {len(states)} rows, expected {expected["rows"]}')
    return states, labels, questions, actual_sha256


def evaluate(agent, states, gold_labels, questions, batch_size):
    import numpy as np
    from laya import ece_score

    matrix = [[0] * len(LABELS) for _ in LABELS]
    nll_sum = 0.0
    brier_sum = 0.0
    confidences = []
    correctness = []
    chunk_size = max(256, batch_size * 8)
    started = time.monotonic()

    for start in range(0, len(states), chunk_size):
        stop = min(start + chunk_size, len(states))
        results = agent.predict_batch(
            states[start:stop], questions, batch_size=batch_size, sort_by_length=True,
        )
        for result, gold_index in zip(results, gold_labels[start:stop]):
            answer = result['answers']['sentiment']
            predicted_label = answer['choice']
            if predicted_label not in LABELS:
                raise ValueError(f'Model returned unexpected label {predicted_label!r}')
            probabilities = np.asarray(
                [answer['probabilities'][label] for label in LABELS], dtype=np.float64,
            )
            probability_sum = float(probabilities.sum())
            if probability_sum <= 0.0 or not np.isfinite(probabilities).all():
                raise ValueError(f'Model returned invalid probabilities {probabilities.tolist()}')
            probabilities /= probability_sum
            predicted_index = int(probabilities.argmax())
            matrix[gold_index][predicted_index] += 1
            nll_sum -= math.log(max(float(probabilities[gold_index]), 1e-12))
            brier_sum += sum(
                (float(probability) - float(index == gold_index)) ** 2
                for index, probability in enumerate(probabilities)
            )
            confidence = float(probabilities[predicted_index])
            confidences.append(confidence)
            correctness.append(predicted_index == gold_index)
        print(f'PROGRESS {stop}/{len(states)}', file=sys.stderr, flush=True)

    per_class = {}
    for index, label in enumerate(LABELS):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        per_class[label] = {
            'support': support,
            'precision': true_positive / max(1, predicted_count),
            'recall': true_positive / max(1, support),
            'f1': 2 * true_positive / max(1, support + predicted_count),
        }
    rows = len(gold_labels)
    correct = sum(matrix[index][index] for index in range(len(LABELS)))
    return {
        'rows': rows,
        'accuracy': correct / rows,
        'macro_f1': sum(item['f1'] for item in per_class.values()) / len(LABELS),
        'nll': nll_sum / rows,
        'brier': brier_sum / rows,
        'ece': float(ece_score(np.asarray(confidences), np.asarray(correctness))),
        'per_class': per_class,
        'confusion_matrix': matrix,
        'label_order': list(LABELS),
        'elapsed_seconds': time.monotonic() - started,
        'rows_per_second': rows / max(time.monotonic() - started, 1e-9),
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    args = arguments()
    if not args.model.is_dir():
        raise FileNotFoundError(
            f'Finished model not found: {args.model}. Wait for training to create '
            'laya-sentiment-model-cap100-natural/final, or pass --model explicitly.'
        )
    if not args.data.is_file():
        raise FileNotFoundError(f'Test set not found: {args.data}')
    if not args.manifest.is_file():
        raise FileNotFoundError(f'Dataset manifest not found: {args.manifest}')

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    try:
        expected = manifest['files'][args.data.name]
    except KeyError as error:
        raise ValueError(f'{args.data.name} is not listed in {args.manifest}') from error
    states, gold_labels, questions, dataset_sha256 = load_dataset(args.data, expected, args.limit)

    from laya import load
    agent = load(str(args.model.resolve()), device=args.device)
    metric_values = evaluate(agent, states, gold_labels, questions, args.batch_size)
    report = {
        'model': str(args.model.resolve()),
        'requested_device': args.device,
        'actual_device': str(agent.device),
        'training': agent.cfg.get('training'),
        'temperature': agent.cfg.get('temperature'),
        'temperature_by_options': agent.cfg.get('temperature_by_options'),
        'dataset': {
            'path': str(args.data.resolve()),
            'sha256': dataset_sha256,
            'manifest_rows': expected['rows'],
            'evaluated_rows': len(states),
            'limit': args.limit,
        },
        'metrics': metric_values,
        'probability_source': 'Laya public predict_batch output, renormalized after 4-decimal serialization',
        'label_provenance': 'Existing machine-generated sentiment labels, not human gold labels',
    }
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

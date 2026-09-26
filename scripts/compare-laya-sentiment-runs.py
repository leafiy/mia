#!/usr/bin/env python3
"""Compare retained prior-best and current Laya runs on the development split.

Both default runs share the fixed 5,000-row development split. The unseen test
set is intentionally excluded: choose a run here first, then run
`test-laya-sentiment.py` exactly once for that selected model.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = (
    ROOT / 'laya-sentiment-model-cap100-natural',
    ROOT / 'laya-sentiment-model-neutral-reviewed-no-uncertain',
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='*', type=Path, default=DEFAULT_RUNS,
                        help='Training output directories (default: prior-best cap100-natural and current neutral-reviewed-no-uncertain)')
    return parser.parse_args()


def main():
    args = arguments()
    rows = []
    expected_development_set = None
    for run in args.runs:
        metrics_path = run / 'metrics.json'
        config_path = run / 'training-config.json'
        history_path = run / 'history.json'
        if not metrics_path.is_file() or not config_path.is_file() or not history_path.is_file():
            raise FileNotFoundError(f'Incomplete training run: {run}')
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        config = json.loads(config_path.read_text(encoding='utf-8'))
        history = json.loads(history_path.read_text(encoding='utf-8'))
        manifest = config['data_manifest']
        development_set = manifest['files']['test.jsonl']
        identity = (development_set['rows'], development_set['sha256'])
        if expected_development_set is None:
            expected_development_set = identity
        elif identity != expected_development_set:
            raise ValueError(
                f'{run}: development split {identity} differs from {expected_development_set}'
            )
        result = metrics['test_after']
        rows.append({
            'run': run.name,
            'train_rows': manifest['files']['train.jsonl']['rows'],
            'sampling': manifest['training_sampling'],
            'accuracy': result['accuracy'],
            'macro_f1': result['macro_f1'],
            'nll': result['nll'],
            'brier': result['brier'],
            'ece': result['ece'],
            'final_train_loss': history[-1]['train_loss'],
        })

    rows.sort(key=lambda row: row['macro_f1'], reverse=True)
    print(f'DEVELOPMENT_SET rows={expected_development_set[0]} sha256={expected_development_set[1]}')
    print('RUN\tTRAIN_ROWS\tACCURACY\tMACRO_F1\tNLL\tBRIER\tECE\tTRAIN_LOSS')
    for row in rows:
        print(
            f'{row["run"]}\t{row["train_rows"]}\t{row["accuracy"]:.6f}\t'
            f'{row["macro_f1"]:.6f}\t{row["nll"]:.6f}\t{row["brier"]:.6f}\t'
            f'{row["ece"]:.6f}\t{row["final_train_loss"]:.6f}'
        )
    print('BEST_BY_MACRO_F1', rows[0]['run'])


if __name__ == '__main__':
    main()

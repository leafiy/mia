#!/usr/bin/env python3
"""Build the reviewed no-uncertain Laya sentiment training dataset.

A neutral row is relabeled only when two independent passes agree on the SAME
polar label. First-pass ambiguity and second-pass disagreement are always
excluded from training. Calibration, development and unseen-test remain
untouched; machine-generated labels are not human gold labels.

Default output names the retained reviewed no-uncertain dataset and refuses to
overwrite it; pass --output with a fresh directory when reproducing:
  python build-laya-neutral-reviewed.py \
    --output laya-sentiment-data-cap100-natural-neutral-reviewed-no-uncertain-reproduction
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ('负面', '中性', '正面')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=ROOT / 'laya-sentiment-data-cap100-natural')
    parser.add_argument('--first', type=Path,
                        default=ROOT / 'laya-sentiment-neutral-review' / 'first-pass.jsonl')
    parser.add_argument('--second', type=Path,
                        default=ROOT / 'laya-sentiment-neutral-review' / 'second-pass.jsonl')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'laya-sentiment-data-cap100-natural-neutral-reviewed-no-uncertain')
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def decisions(path, key):
    result = {}
    with path.open(encoding='utf-8') as source:
        for number, line in enumerate(source, 1):
            item = json.loads(line)
            if item['id'] in result:
                raise ValueError(f'{path}:{number}: repeated id {item["id"]}')
            if item[key] not in (*LABELS, '存疑'):
                raise ValueError(f'{path}:{number}: invalid {key}={item[key]!r}')
            result[item['id']] = item[key]
    return result


def main():
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite output: {args.output}')
    manifest_path = args.source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    first = decisions(args.first, 'first_pass')
    second = decisions(args.second, 'second_pass')
    source_train = args.source / 'train.jsonl'
    expected = manifest['files']['train.jsonl']
    source_train_sha256 = expected['sha256']
    if sha256_file(source_train) != expected['sha256']:
        raise ValueError('Source training JSONL differs from its manifest')
    proposed = {id for id, choice in first.items() if choice in ('正面', '负面')}
    if set(second) != proposed:
        raise ValueError('Second pass must cover every proposed positive/negative change exactly once')
    uncertain_ids = {
        id for id, choice in first.items()
        if choice == '存疑' or (id in proposed and second[id] != choice)
    }
    excluded_ids = uncertain_ids

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.laya-neutral-reviewed-', dir=args.output.parent) as tmp:
        temp = Path(tmp)
        neutral_ids, changes = set(), {}
        labels, outcomes = Counter(), Counter()
        with (source_train.open(encoding='utf-8') as original,
              (temp / 'train.jsonl').open('w', encoding='utf-8') as target,
              (temp / 'neutral-label-audit.jsonl').open('w', encoding='utf-8') as audit):
            for line_number, line in enumerate(original, 1):
                row = json.loads(line)
                gold = json.loads(row['gold'])
                label = gold['sentiment']['label']
                if label == '中性':
                    id = row['id']
                    if id in neutral_ids or id not in first:
                        raise ValueError(f'{source_train}:{line_number}: duplicate or unreviewed neutral id {id}')
                    neutral_ids.add(id)
                    primary = first[id]
                    confirmation = second.get(id)
                    changed = primary in ('正面', '负面') and primary == confirmation
                    excluded = id in excluded_ids
                    final = None if excluded else (primary if changed else label)
                    status = 'confirmed_relabel' if changed else (
                        'ambiguous_first_pass' if primary == '存疑' else
                        'kept_neutral_first_pass' if primary == '中性' else 'disagreed_second_pass'
                    )
                    if excluded:
                        status = 'excluded_' + status
                    outcomes[status] += 1
                    if changed:
                        changes[id] = final
                        gold['sentiment']['label'] = final
                        gold['sentiment']['probabilities'] = {
                            name: float(name == final) for name in LABELS
                        }
                        row['gold'] = json.dumps(gold, ensure_ascii=False)
                        line = json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                    audit.write(json.dumps({
                        'id': id,
                        'text': json.loads(row['state'])['text'],
                        'original': label,
                        'first_pass': primary,
                        'second_pass': confirmation,
                        'final': final,
                        'status': status,
                    }, ensure_ascii=False, separators=(',', ':')) + '\n')
                    if not excluded:
                        labels[final] += 1
                        target.write(line)
                else:
                    if row['id'] in first:
                        raise ValueError(f'{source_train}:{line_number}: non-neutral id was reviewed')
                    labels[label] += 1
                    target.write(line)
        if neutral_ids != set(first):
            raise ValueError('First-pass IDs differ from the original neutral training IDs')
        if sum(labels.values()) != expected['rows'] - len(excluded_ids):
            raise ValueError('Training row count differs from the expected exclusions')

        output_files = manifest['files']
        for name in ('calibration.jsonl', 'test.jsonl', 'unseen-test.jsonl'):
            source = args.source / name
            if sha256_file(source) != output_files[name]['sha256']:
                raise ValueError(f'{name}: source differs from its manifest')
            shutil.copyfile(source, temp / name)
        csv_name = next(name for name in output_files if name.endswith('.csv'))
        source_csv = args.source / csv_name
        if sha256_file(source_csv) != output_files[csv_name]['sha256']:
            raise ValueError('Source CSV differs from its manifest')
        csv_changes, csv_exclusions = set(), set()
        train_industries, train_groups = Counter(), set()
        csv_rows = 0
        with (source_csv.open(newline='', encoding='utf-8') as source,
              (temp / csv_name).open('w', newline='', encoding='utf-8') as target):
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row['id'] in excluded_ids:
                    if row['split'] != 'train' or row['label'] != '中性':
                        raise ValueError(f'Invalid CSV row for exclusion: {row["id"]}')
                    csv_exclusions.add(row['id'])
                    continue
                if row['id'] in changes:
                    if row['split'] != 'train' or row['label'] != '中性':
                        raise ValueError(f'Invalid CSV row for relabel: {row["id"]}')
                    row['label'] = changes[row['id']]
                    csv_changes.add(row['id'])
                if row['split'] == 'train':
                    train_industries[row['industry']] += 1
                    train_groups.add(row['source_group'])
                writer.writerow(row)
                csv_rows += 1
        if csv_changes != set(changes) or csv_exclusions != excluded_ids:
            raise ValueError('CSV change/exclusion IDs differ from training JSONL')
        if csv_rows != output_files[csv_name]['rows'] - len(excluded_ids):
            raise ValueError('CSV row count differs from the expected exclusions')

        output_files['train.jsonl']['sha256'] = sha256_file(temp / 'train.jsonl')
        output_files['train.jsonl']['rows'] = sum(labels.values())
        output_files['train.jsonl']['labels'] = dict(labels)
        output_files['train.jsonl']['industries'] = dict(train_industries)
        output_files['train.jsonl']['source_groups'] = len(train_groups)
        output_files[csv_name]['sha256'] = sha256_file(temp / csv_name)
        output_files[csv_name]['rows'] = csv_rows
        manifest['split_targets']['train'] = sum(labels.values())
        manifest['fields_used_as_text_or_target']['target'] = 'original sentiment labels, with independently confirmed neutral-label semantic corrections'
        manifest['training_sampling'] = (
            f'Same capped source-pool rows as cap100-natural with {len(changes):,} agreed '
            f'neutral-label corrections and {len(excluded_ids):,} uncertain/disputed neutral rows excluded'
        )
        manifest['neutral_label_review'] = {
            'source_train_sha256': source_train_sha256,
            'source_manifest_sha256': sha256_file(manifest_path),
            'first_pass': 'model-assisted contextual sentiment classification without the original label',
            'second_pass': 'independent model-assisted contextual sentiment classification on proposed polar relabels',
            'policy': (
                'Relabel only if both passes agree on the same non-neutral label; '
                'exclude first-pass uncertain and second-pass disputed rows'
            ),
            'machine_generated_not_human_gold': True,
            'reviewed_neutral_rows': len(neutral_ids),
            'proposed_relabels': len(proposed),
            'confirmed_relabels': len(changes),
            'excluded_uncertain_rows': len(excluded_ids),
            'dispositions': dict(outcomes),
            'audit_file': 'neutral-label-audit.jsonl',
            'audit_sha256': sha256_file(temp / 'neutral-label-audit.jsonl'),
        }
        (temp / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temp, args.output)
    print(json.dumps({
        'output': str(args.output),
        'original_train_sha256': manifest['neutral_label_review']['source_train_sha256'],
        'new_train_sha256': output_files['train.jsonl']['sha256'],
        'train_rows': output_files['train.jsonl']['rows'],
        'labels': dict(labels),
        'review': manifest['neutral_label_review'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

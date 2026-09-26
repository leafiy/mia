#!/usr/bin/env python3
"""Clean, deduplicate, group-split, and export Laya sentiment training data.

Default output writes a fresh reproduction directory for the retained
cap100-natural recipe:
  python prepare-laya-sentiment.py

This keeps the SHA-256 source-group split, cleaning, deduplication, train group
cap 100, natural train sampling, and full capped train pool (train rows 0) used
by laya-sentiment-data-cap100-natural, without overwriting that saved dataset.

Holdout groups remain capped at 100 examples each; calibration/test stay fixed
at 5,000 rows, and unseen-test is the remaining test-pool rows.
"""
import argparse
import csv
import hashlib
import html
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent
DEFAULT_SOURCE = ARCHIVE / 'sentiment-training-industry.csv'
DEFAULT_OUTPUT_DIR = ARCHIVE / 'laya-sentiment-data-reproduction'
DEFAULT_TRAIN_ROWS = 0
DEFAULT_TRAIN_GROUP_CAP = 100
DEFAULT_TRAIN_SAMPLING = 'natural'
HOLDOUT_GROUP_CAP = 100
HOLDOUT_TARGETS = {'calibration': 5_000, 'test': 5_000}
LABELS = ('负面', '中性', '正面')
MIN_CHARS = 5
MAX_CHARS = 100
KEEP_PUNCTUATION = set('，。！？；：,.!?;:\'-’')

TAG_RE = re.compile(r'<[^>]*>')
URL_RE = re.compile(r'(?i)(?:https?://|www\.)\S+')
IMAGE_FORMULA_RE = re.compile(r'(?i)=\s*(?:DISPIMG|IMAGE)\s*\([^)]*\)')
MENTION_RE = re.compile(r'(?<![\w.])@[\w.-]+')
WEIBO_FORWARD_RE = re.compile(r'//@[\w.-]+:?')
XHS_EMOJI_RE = re.compile(r'\[[^\[\]\s]{1,12}R\]')
CHINESE_SPACE_RE = re.compile(r'(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff0-9])|(?<=\d)\s+(?=[\u3400-\u9fff])')
REPEATED_PUNCTUATION_RE = re.compile(r'([，。！？；：,.!?;:])\1+')
SPACE_BEFORE_PUNCTUATION_RE = re.compile(r'\s+([，。！？；：,.!?;:])')

EDGE_PUNCTUATION_RE = re.compile(r"^[\s，；：,;:'’\-]+|[\s，；：,;:'’\-]+$")

def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--train-group-cap', type=int, default=DEFAULT_TRAIN_GROUP_CAP,
                        help='Maximum unique rows retained per training source group')
    parser.add_argument('--train-sampling', choices=('balanced', 'natural'), default=DEFAULT_TRAIN_SAMPLING,
                        help='Balance labels equally or preserve the source-pool distribution')
    parser.add_argument('--train-rows', type=int, default=DEFAULT_TRAIN_ROWS,
                        help='Training rows to select; 0 uses every row allowed by the mode and cap')
    args = parser.parse_args()
    if args.train_group_cap < 1:
        parser.error('--train-group-cap must be positive')
    if args.train_rows < 0:
        parser.error('--train-rows must be nonnegative')
    return args


def normalize_text(value):
    text = unicodedata.normalize('NFKC', html.unescape(value or ''))
    text = text.replace('\ufe0e', '').replace('\ufe0f', '')
    text = IMAGE_FORMULA_RE.sub(' ', text)
    text = TAG_RE.sub(' ', text)
    text = URL_RE.sub(' ', text)
    text = WEIBO_FORWARD_RE.sub(' ', text)
    text = MENTION_RE.sub(' ', text)
    text = XHS_EMOJI_RE.sub(' ', text)
    text = re.sub(r'\[话题\]', ' ', text)

    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if category in {'Cc', 'Cf', 'Cs', 'Co', 'Cn'}:
            cleaned.append(' ')
        elif category.startswith('S') and category != 'Sc':
            cleaned.append(' ')
        elif category.startswith('P') and char not in KEEP_PUNCTUATION:
            cleaned.append(' ')
        else:
            cleaned.append(char)

    text = ''.join(cleaned)
    text = re.sub(r"(?<![A-Za-z])['’]|['’](?![A-Za-z])", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = CHINESE_SPACE_RE.sub('', text)
    text = SPACE_BEFORE_PUNCTUATION_RE.sub(r'\1', text)
    text = REPEATED_PUNCTUATION_RE.sub(r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return EDGE_PUNCTUATION_RE.sub('', text).strip()


def stable_hash(value):
    return hashlib.sha256(f'{SEED}|{value}'.encode('utf-8')).hexdigest()


def group_split(group_id):
    bucket = int(stable_hash(f'group|{group_id}')[:8], 16) % 10
    return 'train' if bucket < 8 else ('calibration' if bucket == 8 else 'test')


def allocate_quotas(total, weights):
    weight_sum = sum(weights.values())
    if total > weight_sum:
        raise ValueError(f'Requested {total} samples from a pool of {weight_sum}')
    exact = {key: total * weight / weight_sum for key, weight in weights.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = sorted(weights, key=lambda key: (-(exact[key] - quotas[key]), str(key)))
    for key in order[:remainder]:
        quotas[key] += 1
    if any(quotas[key] > weights[key] for key in quotas):
        raise ValueError('Stratified allocation exceeds available rows')
    return quotas


def load_candidates(database, stats, source_path=DEFAULT_SOURCE,
                    train_group_cap=DEFAULT_TRAIN_GROUP_CAP):
    connection = sqlite3.connect(database)
    connection.executescript('''
        CREATE TABLE rows (
            source_row INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            label TEXT NOT NULL,
            source_group TEXT NOT NULL,
            industry TEXT NOT NULL
        );
        CREATE INDEX rows_text_idx ON rows(text);
        CREATE INDEX rows_group_idx ON rows(source_group);
        CREATE TABLE candidates (
            source_row INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            label TEXT NOT NULL,
            source_group TEXT NOT NULL,
            industry TEXT NOT NULL,
            split TEXT NOT NULL,
            sample_hash TEXT NOT NULL
        );
        CREATE INDEX candidate_group_idx ON candidates(split, source_group, sample_hash);
    ''')
    batch = []
    with source_path.open(newline='', encoding='utf-8') as source:
        reader = csv.DictReader(source)
        required = {'state', 'label', 'source_group', 'industry'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f'Unexpected input CSV columns: {reader.fieldnames}')
        for source_row, row in enumerate(reader, start=2):
            stats['rows_seen'] += 1
            label = (row.get('label') or '').strip()
            if label not in LABELS:
                stats['dropped_label'] += 1
                continue
            source_group = (row.get('source_group') or '').strip()
            if not source_group:
                stats['dropped_missing_source_group'] += 1
                continue
            text = normalize_text(row.get('state', ''))
            if text != (row.get('state') or ''):
                stats['normalized_rows'] += 1
            if not MIN_CHARS <= len(text) <= MAX_CHARS:
                stats['dropped_cleaned_length'] += 1
                continue
            if not any(char.isalnum() for char in text):
                stats['dropped_no_alphanumeric_text'] += 1
                continue
            industry = (row.get('industry') or '').strip() or '其他'
            batch.append((source_row, text, label, source_group, industry))
            stats['eligible_before_dedup'] += 1
            if len(batch) >= 5000:
                connection.executemany('INSERT INTO rows VALUES(?,?,?,?,?)', batch)
                connection.commit()
                batch.clear()
    if batch:
        connection.executemany('INSERT INTO rows VALUES(?,?,?,?,?)', batch)
        connection.commit()

    conflict_stats = connection.execute('''
        SELECT COUNT(*), COALESCE(SUM(row_count), 0)
        FROM (
            SELECT text, COUNT(*) AS row_count
            FROM rows
            GROUP BY text
            HAVING COUNT(DISTINCT label) > 1
        )
    ''').fetchone()
    stats['conflicting_normalized_texts_dropped'] = conflict_stats[0]
    stats['conflicting_rows_dropped'] = conflict_stats[1]
    duplicate_rows = connection.execute('''
        SELECT COALESCE(SUM(row_count - 1), 0)
        FROM (
            SELECT COUNT(*) AS row_count
            FROM rows
            GROUP BY text
            HAVING COUNT(DISTINCT label) = 1
        )
    ''').fetchone()[0]
    stats['same_label_duplicate_rows_dropped'] = duplicate_rows

    unique_rows = connection.execute('''
        SELECT r.source_row, r.text, r.label, r.source_group, r.industry
        FROM rows AS r
        JOIN (
            SELECT MIN(source_row) AS source_row
            FROM rows
            GROUP BY text
            HAVING COUNT(DISTINCT label) = 1
        ) AS keep ON keep.source_row = r.source_row
        ORDER BY r.source_row
    ''')
    batch = []
    for source_row, text, label, source_group, industry in unique_rows:
        split = group_split(source_group)
        batch.append((source_row, text, label, source_group, industry, split,
                      stable_hash(f'cap|{source_row}')))
        stats[f'unique_{split}_rows'] += 1
        if len(batch) >= 5000:
            connection.executemany('INSERT INTO candidates VALUES(?,?,?,?,?,?,?)', batch)
            connection.commit()
            batch.clear()
    if batch:
        connection.executemany('INSERT INTO candidates VALUES(?,?,?,?,?,?,?)', batch)
        connection.commit()

    capped = connection.execute('''
        SELECT source_row, text, label, source_group, industry, split
        FROM (
            SELECT source_row, text, label, source_group, industry, split,
                   ROW_NUMBER() OVER (
                       PARTITION BY split, source_group ORDER BY sample_hash, source_row
                   ) AS group_rank
            FROM candidates
        )
        WHERE group_rank <= CASE WHEN split = 'train' THEN ? ELSE ? END
        ORDER BY split, source_group, source_row
    ''', (train_group_cap, HOLDOUT_GROUP_CAP))
    rows = list(capped)
    connection.close()
    return rows


def select_split(rows, split, train_sampling=DEFAULT_TRAIN_SAMPLING,
                 train_rows=DEFAULT_TRAIN_ROWS):
    pool = [row for row in rows if row[5] == split]
    if split == 'train':
        by_label = defaultdict(list)
        for row in pool:
            by_label[row[2]].append(row)
        if train_rows:
            target = train_rows
        elif train_sampling == 'balanced':
            target = min(len(by_label[label]) for label in LABELS) * len(LABELS)
        else:
            target = len(pool)
        if train_sampling == 'balanced':
            selected = []
            train_base, remainder = divmod(target, len(LABELS))
            label_targets = {label: train_base for label in LABELS}
            for label in LABELS[:remainder]:
                label_targets[label] += 1
            for label in LABELS:
                by_industry = defaultdict(list)
                for row in by_label[label]:
                    by_industry[row[4]].append(row)
                quotas = allocate_quotas(
                    label_targets[label],
                    {industry: len(items) for industry, items in by_industry.items()},
                )
                for industry, quota in quotas.items():
                    selected.extend(sorted(
                        by_industry[industry],
                        key=lambda row: stable_hash(f'pick|{split}|{row[0]}'),
                    )[:quota])
        else:
            strata = defaultdict(list)
            for row in pool:
                strata[(row[2], row[4])].append(row)
            quotas = allocate_quotas(
                target, {stratum: len(items) for stratum, items in strata.items()},
            )
            selected = []
            for stratum, quota in quotas.items():
                selected.extend(sorted(
                    strata[stratum],
                    key=lambda row: stable_hash(f'pick|{split}|{row[0]}'),
                )[:quota])
    else:
        target = HOLDOUT_TARGETS[split]
        strata = defaultdict(list)
        for row in pool:
            strata[(row[2], row[4])].append(row)
        quotas = allocate_quotas(
            target, {stratum: len(items) for stratum, items in strata.items()},
        )
        selected = []
        for stratum, quota in quotas.items():
            selected.extend(sorted(
                strata[stratum],
                key=lambda row: stable_hash(f'pick|{split}|{row[0]}'),
            )[:quota])
    if len(selected) != target:
        raise ValueError(f'{split}: selected {len(selected)} rows, expected {target}')
    return sorted(selected, key=lambda row: stable_hash(f'order|{row[0]}'))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_laya_jsonl(path, rows, question, criteria):
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w', encoding='utf-8') as target:
        for source_row, text, label, _source_group, _industry, _split in rows:
            probabilities = {name: float(name == label) for name in criteria}
            example = {
                'id': f'sent-{source_row:06d}',
                'state': json.dumps({'text': text}, ensure_ascii=False),
                'questions': json.dumps(question, ensure_ascii=False),
                'gold': json.dumps({
                    'sentiment': {
                        'type': 'choice', 'label': label, 'probabilities': probabilities,
                    }
                }, ensure_ascii=False),
            }
            target.write(json.dumps(example, ensure_ascii=False, separators=(',', ':')) + '\n')
    os.replace(temporary, path)
    return {
        'rows': len(rows), 'sha256': sha256_file(path),
        'labels': dict(Counter(row[2] for row in rows)),
        'industries': dict(Counter(row[4] for row in rows)),
        'source_groups': len({row[3] for row in rows}),
    }


def write_outputs(selected, unseen_test, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    criteria = {
        '负面': '表达不满、失望、批评等负面态度',
        '中性': '态度客观或情绪不明显',
        '正面': '表达满意、赞赏、愉快等正面态度',
    }
    question = {
        'sentiment': {
            'type': 'choice',
            'instructions': '判断文本表达的整体情感倾向。',
            'criteria': criteria,
        }
    }
    csv_name = 'sentiment-50k-clean.csv' if len(selected) == 50_000 else 'sentiment-selected-clean.csv'
    csv_path = output_dir / csv_name
    temporary_csv = csv_path.with_suffix('.csv.partial')
    csv_fields = ('id', 'split', 'source_group', 'industry', 'text', 'label')
    with temporary_csv.open('w', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=csv_fields)
        writer.writeheader()
        for source_row, text, label, source_group, industry, split in selected:
            writer.writerow({
                'id': f'sent-{source_row:06d}', 'split': split, 'source_group': source_group,
                'industry': industry, 'text': text, 'label': label,
            })
    os.replace(temporary_csv, csv_path)

    outputs = {}
    for split in ('train', 'calibration', 'test'):
        filename = f'{split}.jsonl'
        split_rows = [row for row in selected if row[5] == split]
        outputs[filename] = write_laya_jsonl(
            output_dir / filename, split_rows, question, criteria,
        )
    outputs['unseen-test.jsonl'] = write_laya_jsonl(
        output_dir / 'unseen-test.jsonl', unseen_test, question, criteria,
    )
    outputs[csv_path.name] = {'rows': len(selected), 'sha256': sha256_file(csv_path)}
    return outputs


def main():
    args = arguments()
    source_path = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f'Missing industry-labeled source CSV: {source_path}')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output directory: {output_dir}')
    stats = Counter()
    source_hash = sha256_file(source_path)
    with tempfile.TemporaryDirectory(prefix='.laya-sentiment-', dir=ARCHIVE) as temp_dir:
        capped_rows = load_candidates(
            Path(temp_dir) / 'selection.sqlite', stats, source_path, args.train_group_cap,
        )
    split_names = ('train', 'calibration', 'test')
    pools = {split: [row for row in capped_rows if row[5] == split] for split in split_names}
    candidate_counts = {
        split: {
            'rows_after_source_group_cap': len(pools[split]),
            'labels': dict(Counter(row[2] for row in pools[split])),
            'industries': dict(Counter(row[4] for row in pools[split])),
        }
        for split in split_names
    }
    selected_by_split = {
        'train': select_split(
            capped_rows, 'train', args.train_sampling, args.train_rows,
        ),
        'calibration': select_split(capped_rows, 'calibration'),
        'test': select_split(capped_rows, 'test'),
    }
    selected = [row for split in split_names for row in selected_by_split[split]]
    selected_test_ids = {row[0] for row in selected_by_split['test']}
    unseen_test = [
        row for row in pools['test']
        if row[0] not in selected_test_ids
    ]
    if len(unseen_test) + len(selected_test_ids) != len(pools['test']):
        raise ValueError('Unseen test rows do not exactly complement the fixed test selection')
    group_sets = {
        split: {row[3] for row in selected_by_split[split]}
        for split in split_names
    }
    if any(group_sets[a] & group_sets[b] for a, b in (
        ('train', 'calibration'), ('train', 'test'), ('calibration', 'test'),
    )):
        raise ValueError('Source-group leakage across dataset splits')
    outputs = write_outputs(selected, unseen_test, output_dir)
    split_targets = {
        split: len(selected_by_split[split])
        for split in split_names
    }
    if args.train_sampling == 'balanced':
        training_sampling = (
            f'{split_targets["train"]:,} rows balanced equally across the three labels; '
            'industry proportions retained within each label'
        )
    else:
        training_sampling = (
            f'{split_targets["train"]:,} rows preserving the capped source-pool '
            'label-by-industry proportions'
        )
    manifest = {
        'format': 'Laya notebook-compatible JSONL; state, questions, and gold are JSON-encoded strings',
        'recommended_base_model': 'convaiinnovations/laya-multilingual',
        'source_file': source_path.name,
        'source_sha256': source_hash,
        'seed': SEED,
        'split_method': 'SHA-256 assignment by source_group: 80% train pool, 10% calibration pool, 10% test pool; no group crosses splits',
        'max_rows_per_source_group_before_sampling': {
            'train': args.train_group_cap,
            'calibration': HOLDOUT_GROUP_CAP,
            'test': HOLDOUT_GROUP_CAP,
        },
        'split_targets': split_targets,
        'training_sampling': training_sampling,
        'calibration_and_test_sampling': '5,000 rows each, stratified to retain their source-pool label-by-industry proportions',
        'unseen_test_sampling': 'All eligible test-pool rows left after the fixed 5,000-row test selection; no row appears in train, calibration, or test.',
        'fields_used_as_text_or_target': {'input_state': 'normalized state text', 'target': 'existing sentiment label'},
        'industry_usage': 'Industry labels are used only for sampling and audit metadata; never included in model input or target.',
        'cleaning': {
            'symbols': 'Emoji, variation selectors, decorative symbols, and control/format characters removed; currency symbols retained; common sentence punctuation and English apostrophes/hyphens retained.',
            'whitespace': 'Repeated whitespace collapsed; spaces between adjacent CJK characters removed. Necessary spaces between Latin words are retained.',
            'punctuation': 'Repeated sentence punctuation collapsed to one mark; spaces before sentence punctuation removed.',
            'length': f'Keep {MIN_CHARS}-{MAX_CHARS} Unicode characters after normalization.',
            'deduplication': 'Keep one copy of each normalized text when all copies have the same label; drop every row of normalized texts with conflicting labels.',
        },
        'filter_counts': dict(stats),
        'candidate_pools_after_group_cap': candidate_counts,
        'files': outputs,
    }
    manifest_path = output_dir / 'manifest.json'
    temporary_manifest = manifest_path.with_suffix('.json.partial')
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_manifest, manifest_path)
    print('OUTPUT_DIR', output_dir)
    for filename, details in outputs.items():
        print('FILE', filename, details.get('rows'), details['sha256'])
    print('FILTERS', json.dumps(dict(stats), ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Evaluate a Mia checkpoint on public Chinese sentiment datasets downloaded from the Hugging Face Hub.

  python scripts/evaluate-public-benchmarks.py --model-type qwen --model models/mia-qwen3.5-2b/final --output reports/public/mia-qwen3.5-2b.json
  python scripts/evaluate-public-benchmarks.py --model-type laya --model models/mia-laya/final --output reports/public/mia-laya.json
  python scripts/evaluate-public-benchmarks.py --model-type clm --model <run>/best_head.pt --clm-repo <CLM> --embed-model <Qwen3-8B> \
    --clm-cache <workdir>/embeddings/public.npz --clm-config reports/<alias>/sentiment-config.json --output reports/public/<alias>.json

Datasets (fixed samples, seed 42; texts normalized like training and cut to 512 characters):
  chnsenticorp   lansinuote/ChnSentiCorp test split, 1,200 hotel/book/laptop reviews, binary
  weibo_senti    dirtycomputer/weibo_senti_100k, 5,000 sampled Weibo posts, binary (emoticon-derived labels)
  online_shop    dirtycomputer/online_shopping_10_cats, 5,000 sampled e-commerce reviews across 10 categories, binary
  eprstmt        suolyer/eprstmt (FewCLUE) test split, e-commerce reviews, binary
  dmsc           BerlinWang/DMSC, 5,000 sampled Douban movie short comments with 1-5 stars: 1-2 -> 负面, 3 -> 中性, 4-5 -> 正面
For binary sets a 中性 prediction counts as wrong in `accuracy`; `polarity_accuracy` scores only the rows where the model
chose 正面/负面, and `neutral_rate` is how often it chose 中性. The 3-star proxy for 中性 in dmsc is rough by nature.
"""
import argparse
import csv
import importlib.util
import io
import json
import os
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parent
LABELS = ['负面', '中性', '正面']
SEED = 42
SAMPLE = 5000
MAX_CHARS = 512


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-type', choices=('qwen', 'laya', 'zero-shot', 'clm', 'decider', 'kev'), required=True,
                        help='qwen/laya: a Mia final/ directory; zero-shot: an original Qwen3.5 checkpoint scored by evaluate-qwen-zero-shot.py; '
                             'clm: a CLM head checkpoint scored by evaluate-clm-sentiment.py; decider/kev: a System One checkpoint '
                             '(Hub id or path: models/mia-decider-2b/final, Mapika/decider-2b, jaredpalmer/kev-4b) scored by evaluate-systemone-sentiment.py')
    parser.add_argument('--model', type=Path, required=True, help='final/ directory (or the base checkpoint for zero-shot, the head .pt for clm)')
    parser.add_argument('--questions', type=Path, help='questions.json for laya (default: beside --model)')
    parser.add_argument('--clm-repo', type=Path, help='clm: CLM checkout')
    parser.add_argument('--embed-model', type=Path, help='clm: Qwen3-8B checkpoint directory')
    parser.add_argument('--clm-cache', type=Path, help='clm: TextCache .npz for the benchmark embeddings (shared across heads)')
    parser.add_argument('--clm-config', type=Path, help='clm: sentiment-config.json holding the temperature (default: temperature 1)')
    parser.add_argument('--schema-cache', action='store_true', help='decider: questions-first layout with the question prefix cached')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--datasets', nargs='*', default=['chnsenticorp', 'weibo_senti', 'online_shop', 'eprstmt', 'dmsc'])
    parser.add_argument('--cache-dir', type=Path, default=ROOT.parent / 'data' / 'public-cache')
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def download(repo, filename, cache_dir):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, repo_type='dataset', cache_dir=str(cache_dir)))


def sample(rows, n):
    rows = list(rows)
    random.Random(SEED).shuffle(rows)
    return rows[:n]


def load_dataset(name, cache_dir):
    """Return (rows of (text, gold_label), kind) where kind is 'binary' or 'ternary'."""
    if name == 'chnsenticorp':
        import pandas as pd
        path = download('lansinuote/ChnSentiCorp', 'data/test-00000-of-00001-5372924f059fe767.parquet', cache_dir)
        df = pd.read_parquet(path)
        return [(str(t), '正面' if int(l) == 1 else '负面') for t, l in zip(df['text'], df['label'])], 'binary'
    if name == 'weibo_senti':
        path = download('dirtycomputer/weibo_senti_100k', 'weibo_senti_100k.csv', cache_dir)
        with path.open(newline='', encoding='utf-8') as f:
            rows = [(r['review'], '正面' if r['label'].strip() == '1' else '负面') for r in csv.DictReader(f)]
        return sample(rows, SAMPLE), 'binary'
    if name == 'online_shop':
        path = download('dirtycomputer/online_shopping_10_cats', 'online_shopping_10_cats.csv', cache_dir)
        with path.open(newline='', encoding='utf-8') as f:
            rows = [(r['review'], '正面' if r['label'].strip() == '1' else '负面') for r in csv.DictReader(f) if r['review']]
        return sample(rows, SAMPLE), 'binary'
    if name == 'eprstmt':
        # This Hub copy is instruction-formatted: `input` = fixed instruction + review, `output` = 正面 / 负面.
        path = download('suolyer/eprstmt', 'test.json', cache_dir)
        marker = '则生成答案“负面”。'
        rows = []
        for line in path.open(encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj['input']
            if marker in text:
                text = text.split(marker, 1)[1]
            if obj['output'] not in ('正面', '负面') or not text.strip():
                continue
            rows.append((text.strip(), obj['output']))
        return rows, 'binary'
    if name == 'dmsc':
        path = download('BerlinWang/DMSC', 'DMSC.csv.zip', cache_dir)
        with zipfile.ZipFile(path) as z:
            member = [n for n in z.namelist() if n.lower().endswith('.csv')][0]
            with z.open(member) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8'))
                rows = []
                for r in reader:
                    try:
                        star = int(r['Star'])
                    except (KeyError, ValueError):
                        continue
                    text = r.get('Comment', '')
                    if not text:
                        continue
                    rows.append((text, '负面' if star <= 2 else '中性' if star == 3 else '正面'))
        return sample(rows, SAMPLE), 'ternary'
    raise ValueError(name)


def predict_qwen(model_dir, texts, device, batch_size):
    """Calibrated class probabilities in LABELS order."""
    import torch
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5TextForSequenceClassification
    trainer = module('train-qwen-sentiment.py', 'qwen_sentiment_trainer')
    cfg = json.loads((model_dir / 'sentiment-config.json').read_text(encoding='utf-8'))
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3_5TextForSequenceClassification.from_pretrained(model_dir, dtype=torch.bfloat16).to(device).eval()
    ids = trainer.encode(tok, texts, 8192)
    logits, _ = trainer.collect_logits(model, ids, [0] * len(texts), batch_size, tok.pad_token_id, torch.device(device))
    return (logits / float(cfg['temperature'])).softmax(-1).tolist()


def predict_laya(model_dir, questions_path, texts, device, batch_size):
    """Calibrated class probabilities in LABELS order (laya applies its stored temperature)."""
    from laya import load
    agent = load(str(model_dir), device=device)
    questions = json.loads(questions_path.read_text(encoding='utf-8'))
    out = []
    for start in range(0, len(texts), 512):
        results = agent.predict_batch([{'text': t} for t in texts[start:start + 512]], questions, batch_size=batch_size, sort_by_length=True)
        for r in results:
            p = r['answers']['sentiment']['probabilities']
            total = sum(p.values()) or 1.0
            out.append([p[l] / total for l in LABELS])
    return out


def score(gold, probs, kind):
    """Strict 3-way scoring; for binary sets also the forced two-way choice (argmax over 负面/正面 only)."""
    n = len(gold)
    pred = [LABELS[max(range(3), key=lambda i: p[i])] for p in probs]
    report = {'rows': n, 'gold_counts': dict(Counter(gold)), 'pred_counts': dict(Counter(pred)),
              'accuracy': sum(g == p for g, p in zip(gold, pred)) / n}
    matrix = [[sum(g == a and p == b for g, p in zip(gold, pred)) for b in LABELS] for a in LABELS]
    report['confusion_matrix'] = matrix
    report['label_order'] = LABELS
    per_class = {}
    for i, lab in enumerate(LABELS):
        tp, sup, pc = matrix[i][i], sum(matrix[i]), sum(r[i] for r in matrix)
        per_class[lab] = {'support': sup, 'precision': tp / max(1, pc), 'recall': tp / max(1, sup), 'f1': 2 * tp / max(1, sup + pc)}
    report['per_class'] = per_class
    if kind == 'binary':
        polar = [(g, p) for g, p in zip(gold, pred) if p != '中性']
        report['neutral_rate'] = sum(p == '中性' for p in pred) / n
        report['polarity_accuracy'] = sum(g == p for g, p in polar) / max(1, len(polar))
        forced = ['正面' if p[2] >= p[0] else '负面' for p in probs]
        report['forced_binary_accuracy'] = sum(g == f for g, f in zip(gold, forced)) / n
        f1s = []
        for lab in ('负面', '正面'):
            tp = sum(g == lab and f == lab for g, f in zip(gold, forced))
            sup, pc = sum(g == lab for g in gold), sum(f == lab for f in forced)
            f1s.append(2 * tp / max(1, sup + pc))
        report['forced_binary_macro_f1'] = sum(f1s) / 2
    else:
        report['macro_f1'] = sum(v['f1'] for v in per_class.values()) / 3
    return report


def main():
    args = arguments()
    prep = module('prepare-laya-sentiment.py', 'laya_sentiment_preparation')
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    report = {'model': str(args.model), 'model_type': args.model_type, 'seed': SEED, 'sample': SAMPLE, 'max_chars': MAX_CHARS, 'datasets': {}}
    for name in args.datasets:
        rows, kind = load_dataset(name, args.cache_dir)
        texts = [prep.normalize_text(t)[:MAX_CHARS] or '。' for t, _ in rows]
        gold = [l for _, l in rows]
        if args.model_type == 'qwen':
            probs = predict_qwen(args.model, texts, args.device, args.batch_size)
        elif args.model_type == 'zero-shot':
            if 'zero_shot' not in globals():
                globals()['zero_shot'] = module('evaluate-qwen-zero-shot.py', 'qwen_zero_shot').ZeroShot(args.model, args.device)
            probs = zero_shot.probs(texts, args.batch_size)
        elif args.model_type == 'clm':
            if 'clm_scorer' not in globals():
                globals()['clm_scorer'] = module('evaluate-clm-sentiment.py', 'clm_sentiment').ClmSentiment(
                    args.clm_repo, args.embed_model, args.model, args.device, args.batch_size, args.clm_cache)
            temperature = json.loads(args.clm_config.read_text(encoding='utf-8'))['temperature'] if args.clm_config else 1.0
            probs = (clm_scorer.logits(texts) / temperature).softmax(-1).tolist()
        elif args.model_type in ('decider', 'kev'):
            if 'systemone_scorer' not in globals():
                globals()['systemone_scorer'] = module('evaluate-systemone-sentiment.py', 'systemone_sentiment').scorer(
                    args.model_type, str(args.model), args.device, args.batch_size, schema_cache=args.schema_cache)
            probs = systemone_scorer.probs(texts)
        else:
            probs = predict_laya(args.model, args.questions or args.model.parent / 'questions.json', texts, args.device, args.batch_size)
        res = score(gold, probs, kind)
        res['kind'] = kind
        res['avg_chars'] = sum(len(t) for t in texts) / len(texts)
        report['datasets'][name] = res
        extra = (f"neutral_rate={res['neutral_rate']:.3f} polarity_acc={res['polarity_accuracy']:.4f} forced_binary_acc={res['forced_binary_accuracy']:.4f}"
                 if kind == 'binary' else f"macro_f1={res['macro_f1']:.4f}")
        print(f"{name}: rows={res['rows']} acc={res['accuracy']:.4f} {extra}", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: {kk: v[kk] for kk in ('rows', 'accuracy') } for k, v in report['datasets'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()

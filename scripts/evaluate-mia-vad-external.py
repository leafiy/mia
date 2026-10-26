#!/usr/bin/env python3
"""Score mia-vad-2b on two public Chinese emotion benchmarks that were never used for training or model selection:

  CVAT (github.com/NYCU-NLP/Chinese-EmoBank, ChineseEmoBank/CVAT_SD/CVAT_all_SD.csv): human valence and arousal ratings on 1-9;
      reported as the Pearson r between the ratings and the model's expected level, overall and per text category.
  SMP2020-EWECT, usual (general Weibo) test set, usual_test_labeled.txt: six labels (happy / angry / sad / fear / surprise / neutral);
      reported as accuracy and Macro-F1 of the model's dominant emotion. EWECT has no disgust class: "strict" counts a 厌恶
      answer as wrong, "disgust_as_anger" reads it as 愤怒.

  <decider env>/python scripts/evaluate-mia-vad-external.py --model models/mia-vad-2b --cvat CVAT_all_SD.csv --ewect usual_test_labeled.txt \
      --output reports/mia-vad-2b/external-rerun.json

CVAT_all_SD.csv is tab separated; 38 row numbers repeat and 3 broken bytes glue neighbouring rows together, so rows are keyed by
line number and lines with a broken byte, or without readable valence and arousal, are skipped (2,931 rows remain). Every text gets
the training normalisation (scripts/prepare-laya-sentiment.py normalize_text); texts that end up empty are skipped.
The six questions are models/mia-vad-2b/questions.json, scored in the questions-first layout without CUDA graphs: every
(batch, length) graph would hold its own copy of the six cached question prefixes. Probabilities are taken at T=1, as in
reports/mia-vad-2b/analysis.json; the dominant emotion is the argmax and does not depend on temperature, while the correlations at
the released score temperature (0.88) are 0.839 (valence) and 0.546 (arousal).
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parent
EMOTIONS = ['喜悦', '愤怒', '悲伤', '恐惧', '惊讶', '厌恶', '无明显情绪']
EWECT = {'happy': '喜悦', 'angry': '愤怒', 'sad': '悲伤', 'fear': '恐惧', 'surprise': '惊讶', 'neutral': '无明显情绪'}
CVAT_CATEGORIES = ['news', 'political', 'hotel', 'book', 'car', 'laptop']

_spec = importlib.util.spec_from_file_location('prep', ROOT / 'prepare-laya-sentiment.py')
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def cvat_rows(path):
    """-> [(id, text, valence, arousal, category or None)]"""
    out = []
    for n, line in enumerate(Path(path).read_text(encoding='utf-8', errors='replace').split('\n')[1:], start=1):
        f = line.rstrip('\n').split('\t')
        if len(f) < 6 or '�' in line:
            continue
        has_cat = not f[-1].replace('.', '', 1).isdigit()
        cat = (f[-1].strip() or None) if has_cat else None
        nums = f[-5:-1] if has_cat else f[-4:]
        text = prep.normalize_text('\t'.join(f[1:len(f) - (5 if has_cat else 4)]))
        try:
            v, a = float(nums[0]), float(nums[1])
        except ValueError:
            continue
        if text:
            out.append((f'cvat-L{n}', text, v, a, cat))
    return out


def ewect_rows(path):
    """-> [(id, text, label)]"""
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    out = [(f"ewect-{r['id']}", prep.normalize_text(r['content']), r['label']) for r in rows]
    return [r for r in out if r[1]]


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def expected(probabilities):
    levels = sorted(probabilities, key=float)
    ps = [float(probabilities[k]) for k in levels]
    return sum(float(k) * p for k, p in zip(levels, ps)) / (sum(ps) or 1.0)


def macro_f1(gold, pred, labels):
    f1s = {}
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        f1s[c] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return sum(f1s.values()) / len(f1s), f1s


def answer(schema, texts, batch=64):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        for i, r in zip(idx, schema.batch([texts[i][:512] for i in idx])):
            out[i] = r['answers']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='models/mia-vad-2b (with final/ and questions.json)')
    ap.add_argument('--cvat', required=True)
    ap.add_argument('--ewect', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    from decider.infer import Decider
    from decider.schema_engine import SchemaEngine
    d = Decider(str(Path(args.model) / 'final'), device='cuda', temperature=1.0,
                temperature_by_type={'choice': 1.0, 'score': 1.0, 'noul': 1.0})
    d._se = SchemaEngine(d.eng, use_graphs=False, chat=d.chat)
    schema = d.schema(json.loads((Path(args.model) / 'questions.json').read_text(encoding='utf-8')))

    cv = cvat_rows(args.cvat)
    answers = answer(schema, [r[1] for r in cv])
    cvat = {}
    for cat in [None] + CVAT_CATEGORIES:
        rows = [(r, a) for r, a in zip(cv, answers) if cat is None or r[4] == cat]
        cvat[cat or 'all'] = {'n': len(rows),
                              'valence_r': round(pearson([r[2] for r, _ in rows], [expected(a['valence']['probabilities']) for _, a in rows]), 4),
                              'arousal_r': round(pearson([r[3] for r, _ in rows], [expected(a['arousal']['probabilities']) for _, a in rows]), 4)}

    ew = ewect_rows(args.ewect)
    answers = answer(schema, [r[1] for r in ew])
    gold = [EWECT[r[2]] for r in ew]
    raw = [max(EMOTIONS, key=lambda e: float(a['emotion']['probabilities'][e])) for a in answers]
    ewect = {'n': len(ew), 'predicted_disgust': sum(p == '厌恶' for p in raw)}
    for mode, fix in (('strict', lambda p: p), ('disgust_as_anger', lambda p: '愤怒' if p == '厌恶' else p)):
        pred = [fix(p) for p in raw]
        mf1, per = macro_f1(gold, pred, list(EWECT.values()))
        ewect[mode] = {'acc': round(sum(a == b for a, b in zip(gold, pred)) / len(gold), 4), 'macro_f1': round(mf1, 4),
                       'f1': {k: round(v, 3) for k, v in per.items()}}

    report = {'model': args.model, 'cvat': cvat, 'ewect': ewect}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

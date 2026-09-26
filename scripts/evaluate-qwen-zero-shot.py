#!/usr/bin/env python3
"""Zero-shot baseline: the untouched Qwen3.5 instruct checkpoint asked to pick 负面 / 中性 / 正面.

  python scripts/evaluate-qwen-zero-shot.py --model-dir /path/to/Qwen3.5-2B --data data/eval/unseen-test.jsonl --output reports/qwen3.5-2b-zero-shot/eval-holdout-metrics.json
  python scripts/evaluate-qwen-zero-shot.py --model-dir /path/to/Qwen3.5-2B --csv --output reports/qwen3.5-2b-zero-shot/chinese-csv-evaluation.json
  python scripts/evaluate-qwen-zero-shot.py --model-dir /path/to/Qwen3.5-2B --self-check

The model gets the same three-way rubric Mia was trained with (褒贬并存 counts as 中性), thinking mode off, and is
scored by constrained choice: the next-token logits at the three single-token labels 负面 / 中性 / 正面 are
softmaxed into class probabilities. No calibration, no fine-tuning. Batches are left-padded so the last position
is the answer slot; --self-check compares that against unpadded single-row forwards.
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
LABELS = ['负面', '中性', '正面']
SYSTEM = ('你是中文情感分类器。判断文本作者表达的整体情感倾向，只回答一个词：负面、中性或正面。'
          '正面：表达满意、赞赏、喜爱、愉快、感谢、期待、推荐等态度。'
          '负面：表达不满、失望、批评、愤怒、担忧、厌恶、贬损等态度，反讽按作者实际态度判为负面。'
          '中性：客观陈述、信息说明、提问、叙述、事实报道，作者态度或情绪不明显；褒贬并存、无法归为一方的也算中性。')
TEMPLATE = '文本：{text}\n情感倾向：'


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir', type=Path, required=True, help='Original Qwen3.5 checkpoint directory')
    parser.add_argument('--data', type=Path, help='Manifest-listed JSONL split')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--csv', action='store_true', help='Evaluate the three 中文情感测试集 CSVs')
    parser.add_argument('--self-check', action='store_true', help='Compare left-padded batches with unpadded rows on 48 texts')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if sum(bool(x) for x in (args.data, args.csv, args.self_check)) != 1:
        parser.error('give exactly one of --data, --csv, --self-check')
    if args.data and not args.manifest:
        args.manifest = args.data.parent / 'manifest.json'
    return args


class ZeroShot:
    def __init__(self, model_dir, device):
        import torch
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
        self.torch = torch
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = 'left'
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16).to(self.device).eval()
        self.label_ids = []
        for lab in LABELS:
            ids = self.tok(lab, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                raise ValueError(f'label {lab!r} is not a single token: {ids}')
            self.label_ids.append(ids[0])

    def prompt(self, text):
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': TEMPLATE.format(text=text)}]
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)

    def probs(self, texts, batch_size, padded=True):
        """Class probabilities from the next-token logits at the three label tokens."""
        torch = self.torch
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        result = [None] * len(texts)
        with torch.inference_mode():
            step = batch_size if padded else 1
            for start in range(0, len(order), step):
                idx = order[start:start + step]
                enc = self.tok([self.prompt(texts[i]) for i in idx], return_tensors='pt', padding=padded, add_special_tokens=False).to(self.device)
                try:
                    logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
                except TypeError:
                    logits = self.model(**enc).logits[:, -1, :]
                sub = logits[:, self.label_ids].float().softmax(-1).cpu().tolist()
                for i, p in zip(idx, sub):
                    result[i] = p
        return result


def metrics_from_probs(probs, labels):
    import torch
    trainer = module('train-qwen-sentiment.py', 'qwen_sentiment_trainer')
    logits = torch.tensor(probs, dtype=torch.float32).clamp_min(1e-9).log()
    return trainer.metrics(logits, torch.tensor(labels, dtype=torch.long), 1.0)


def main():
    args = arguments()
    prep = module('prepare-laya-sentiment.py', 'laya_sentiment_preparation')
    zs = ZeroShot(args.model_dir, args.device)
    if args.self_check:
        texts = [json.loads(json.loads(l)['state'])['text'] for l in (ROOT.parent / 'data/eval/test.jsonl').open(encoding='utf-8').readlines()[:48]]
        a = zs.probs(texts, args.batch_size, padded=True)
        b = zs.probs(texts, 1, padded=False)
        agree = sum(max(range(3), key=lambda k: x[k]) == max(range(3), key=lambda k: y[k]) for x, y in zip(a, b))
        diff = max(abs(x[k] - y[k]) for x, y in zip(a, b) for k in range(3))
        print(f'SELF_CHECK argmax agreement {agree}/{len(texts)}, max prob diff {diff:.4f}')
        return
    if args.data:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        expected = manifest['files'][args.data.name]
        digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
        if digest != expected['sha256']:
            raise ValueError(f'{args.data}: SHA-256 differs from manifest')
        texts, labels = [], []
        for line in args.data.open(encoding='utf-8'):
            row = json.loads(line)
            texts.append(json.loads(row['state'])['text'])
            labels.append(LABELS.index(json.loads(row['gold'])['sentiment']['label']))
        m = metrics_from_probs(zs.probs(texts, args.batch_size), labels)
        report = {'model': str(args.model_dir), 'method': 'zero-shot constrained choice over 负面/中性/正面 next-token logits, thinking off, no calibration',
                  'requested_device': args.device, 'temperature': [1.0, 1.0, 1.0],
                  'dataset': {'path': str(args.data), 'sha256': digest, 'manifest_rows': expected['rows'], 'evaluated_rows': len(texts), 'limit': 0},
                  'metrics': m, 'label_provenance': manifest.get('fields_used_as_text_or_target', {}).get('target', 'dataset gold labels')}
        print(f"{args.data.name}: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} neutral_recall={m['per_class']['中性']['recall']:.3f}")
    else:
        trainer = module('train-qwen-sentiment.py', 'qwen_sentiment_trainer')
        report = {'model': str(args.model_dir), 'method': 'zero-shot constrained choice', 'training_data': None, 'labels': LABELS, 'datasets': {}}
        for path in trainer.CSVS:
            rows = []
            with path.open(newline='', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    rows.append((prep.normalize_text(row['text']), row['label'].strip()))
            m = metrics_from_probs(zs.probs([t for t, _ in rows], args.batch_size), [LABELS.index(l) for _, l in rows])
            report['datasets'][path.name] = {'path': str(path), 'label_counts': dict(Counter(l for _, l in rows)), 'exact_train_text_overlap': None, 'metrics': m}
            print(f"{path.name}: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} neutral_recall={m['per_class']['中性']['recall']:.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

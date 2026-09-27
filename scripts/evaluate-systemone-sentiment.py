#!/usr/bin/env python3
"""Score open System One (Jev-style) decision models on Mia's evaluation sets: decider (github.com/Mapika/decider) and Kev
(github.com/jaredpalmer/kev) zero-shot, and mia-decider-2b (decider-2b continued on Mia's data), all asked the same
three-way question as the Qwen3.5 zero-shot baseline.

  <decider env>/python scripts/evaluate-systemone-sentiment.py --impl decider --model Mapika/decider-2b --data data/eval/unseen-test.jsonl --output reports/decider-2b-zero-shot/eval-holdout-metrics.json
  <kev env>/python     scripts/evaluate-systemone-sentiment.py --impl kev --model jaredpalmer/kev-4b --csv --output reports/kev-4b-zero-shot/chinese-csv-evaluation.json
  <decider env>/python scripts/evaluate-systemone-sentiment.py --impl decider --model models/mia-decider-2b/final --data data/eval/unseen-test.jsonl
  ... --throughput --output reports/<alias>/throughput.json
  ... --impl decider --model <fine-tuned model/> --calibrate --output reports/<alias>/calibration-metrics.json

--calibrate (decider) fits one temperature on data/eval/calibration.jsonl from the raw (T=1) probabilities with the Qwen
trainer's golden-section search and writes it into <model>/decider_config.json (temperature and temperature_by_type.choice),
so later runs of this script, and decider's server, answer at it. Use it only on a checkpoint trained on Mia's data;
--method-note then records how that checkpoint was made. --schema-cache (decider) scores in the questions-first layout
with the question block computed once and reused for every text (Decider.schema: CUDA graphs, cached prefix); the
checkpoint must have been trained with that layout (decider.train --schema_first_prob > 0).

Each text is the state (a plain string) and the question is one Choice whose instructions and criteria carry the
rubric of evaluate-qwen-zero-shot.py (褒贬并存 counts as 中性); mia-decider-2b was trained on exactly this question
(models/mia-decider-2b/questions.json). Probabilities are the models' own, at the temperature in the checkpoint's config;
for the zero-shot checkpoints nothing is fitted on Mia's data. decider runs eager (DecisionModel.slot_logits, right-padded
batches); Kev runs in bf16 with the adapter merged (kev.serve's dtype), eager, through DecisionModel.forward_batch.
--throughput times the whole path on the holdout texts: prompt building, tokenisation, forward, softmax.
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
LABELS = ['负面', '中性', '正面']
QUESTIONS = {'sentiment': {
    'type': 'choice',
    'instructions': '判断文本作者表达的整体情感倾向。',
    'criteria': {
        '负面': '表达不满、失望、批评、愤怒、担忧、厌恶、贬损等态度，反讽按作者实际态度判为负面',
        '中性': '客观陈述、信息说明、提问、叙述、事实报道，作者态度或情绪不明显；褒贬并存、无法归为一方的也算中性',
        '正面': '表达满意、赞赏、喜爱、愉快、感谢、期待、推荐等态度'}}}


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DeciderScorer:
    def __init__(self, model, device, batch_size, temperature=None, schema_cache=False):
        import torch
        from decider.infer import Decider
        from decider import temperature as TT
        from decider.model import collate
        self.torch, self.TT, self.collate = torch, TT, collate
        self.d = Decider(model, device=device, temperature=temperature, use_graphs=schema_cache)
        self.schema = self.d.schema(QUESTIONS) if schema_cache else None
        self.batch_size, self.device = batch_size, device

    def probs(self, texts):
        if self.schema is not None:
            return self._schema_probs(texts)
        torch, d = self.torch, self.d
        items = []
        for t in texts:
            rqs, index, its = d._system_one_items(t, QUESTIONS)
            if len(its) != 1 or rqs['sentiment'].get('names', LABELS) != LABELS:
                raise ValueError(f"unexpected prompt rows: {len(its)} rows, names {rqs['sentiment'].get('names')}")
            items.append(its[0])
        order = sorted(range(len(items)), key=lambda i: len(items[i]['ids']))
        out = [None] * len(items)
        with torch.no_grad():
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                part = [items[i] for i in idx]
                b = collate_move(self.collate(part, d.m.tok.pad_token_id), self.device)
                logits = d.m.slot_logits(b['input_ids'], b['attention_mask'], b['slot_idx'], b['slot_batch'], b['nopts'])
                T = self.TT.for_items(d.T, d.T_by_type, part)
                p = self.TT.scaled_softmax(logits, self.TT.slot_temperatures(T, part)).cpu()
                for i, row in zip(idx, p):
                    out[i] = row[:3].tolist()
        return out

    def _schema_probs(self, texts):
        """Questions-first layout with the question block computed once and cached (Decider.schema, CUDA graphs)."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            for i, r in zip(idx, self.schema.batch([texts[i] for i in idx])):
                p = r['answers']['sentiment']['probabilities']
                out[i] = [p[l] for l in LABELS]
        return out

    def describe(self):
        path = ('schema-first layout, question prefix cached once (Decider.schema, CUDA graphs), bf16' if self.schema is not None
                else 'eager DecisionModel.slot_logits, bf16, right-padded batches')
        return {'impl': 'decider', 'model_name': self.d.name, 'temperature': self.d.T, 'temperature_by_type': self.d.T_by_type,
                'layout': self.d.layout, 'path': path}


def collate_move(batch, device):
    return {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}


class KevScorer:
    def __init__(self, model, device, batch_size):
        import torch
        from kev.api import SystemOneRequest, to_record
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE
        self.torch, self.to_record, self.request = torch, to_record, SystemOneRequest
        self.limits = dict(max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
        self.checkpoint = Checkpoint(model)
        self.tok, self.model = self.checkpoint.load(device, LoadOptions(dtype=torch.bfloat16))
        self.model.eval()
        self.batch_size = batch_size

    def probs(self, texts):
        torch = self.torch
        encs = []
        for t in texts:
            rec, meta = self.to_record(self.request.model_validate({'state': t, 'questions': QUESTIONS}))
            if meta[0]['keys'] != LABELS:
                raise ValueError(f'option order {meta[0]["keys"]} differs from {LABELS}')
            encs.append(self.model.encode(self.tok, rec, **self.limits))
        order = sorted(range(len(encs)), key=lambda i: len(encs[i]['ids']))
        out = [None] * len(encs)
        with torch.no_grad():
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                logits = self.model.forward_batch([encs[i] for i in idx])
                for i, per_question in zip(idx, logits):
                    out[i] = torch.softmax(per_question[0].float(), -1).cpu().tolist()
        return out

    def describe(self):
        return {'impl': 'kev', 'checkpoint': str(self.checkpoint.path), 'temperature': float(self.model.head.temperature),
                'path': 'bf16, LoRA merged, eager DecisionModel.forward_batch (row form on the hybrid base)'}


def scorer(impl, model, device, batch_size, temperature=None, schema_cache=False):
    if impl == 'decider':
        return DeciderScorer(model, device, batch_size, temperature, schema_cache)
    if temperature is not None or schema_cache:
        raise ValueError('temperature override and schema cache are only wired for decider')
    return KevScorer(model, device, batch_size)


def metrics_from_probs(probs, labels):
    import torch
    trainer = module('train-qwen-sentiment.py', 'qwen_sentiment_trainer')
    logits = torch.tensor(probs, dtype=torch.float32).clamp_min(1e-9).log()
    return trainer.metrics(logits, torch.tensor(labels, dtype=torch.long), 1.0)


def load_split(path, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = manifest['files'][path.name]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected['sha256']:
        raise ValueError(f'{path}: SHA-256 differs from manifest')
    texts, labels = [], []
    for line in path.open(encoding='utf-8'):
        row = json.loads(line)
        texts.append(json.loads(row['state'])['text'])
        labels.append(LABELS.index(json.loads(row['gold'])['sentiment']['label']))
    if len(texts) != expected['rows']:
        raise ValueError(f'{path}: row count differs from manifest')
    return texts, labels, digest, expected['rows'], manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--impl', choices=('decider', 'kev'), required=True)
    parser.add_argument('--model', required=True, help='Hub id or local checkpoint (Mapika/decider-2b, jaredpalmer/kev-4b)')
    parser.add_argument('--data', type=Path, help='Manifest-listed JSONL split')
    parser.add_argument('--csv', action='store_true', help='Evaluate the three 中文情感测试集 CSVs')
    parser.add_argument('--throughput', action='store_true', help='Time the whole path on data/eval/unseen-test.jsonl')
    parser.add_argument('--calibrate', action='store_true', help='decider: fit the temperature on calibration.jsonl into decider_config.json')
    parser.add_argument('--method-note', help='replaces the zero-shot method text in the reports (for a fine-tuned checkpoint)')
    parser.add_argument('--schema-cache', action='store_true', help='decider: questions-first layout with the question prefix cached (Decider.schema)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if sum(bool(x) for x in (args.data, args.csv, args.throughput, args.calibrate)) != 1:
        parser.error('give exactly one of --data, --csv, --throughput, --calibrate')
    if (args.calibrate or args.schema_cache) and args.impl != 'decider':
        parser.error('--calibrate and --schema-cache are decider only')
    if args.calibrate and args.schema_cache:
        parser.error('calibrate on the default (state-first) path')
    eval_dir = ROOT.parent / 'data' / 'eval'
    s = scorer(args.impl, args.model, args.device, args.batch_size, 1.0 if args.calibrate else None, args.schema_cache)
    method = args.method_note or (f'zero-shot System One ({args.impl}): one Choice with the evaluate-qwen-zero-shot.py rubric, state = the text, '
                                  'probabilities at the checkpoint temperature, nothing fitted on Mia data')

    if args.calibrate:
        import torch
        trainer = module('train-qwen-sentiment.py', 'qwen_sentiment_trainer')
        texts, labels, digest, rows, _ = load_split(eval_dir / 'calibration.jsonl', eval_dir / 'manifest.json')
        logits = torch.tensor(s.probs(texts), dtype=torch.float32).clamp_min(1e-9).log()
        gold = torch.tensor(labels, dtype=torch.long)
        temperature = trainer.fit_temperature(logits, gold)
        config_path = Path(args.model) / 'decider_config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        config['temperature'] = temperature
        config['temperature_by_type'] = {**config.get('temperature_by_type', {}), 'choice': temperature}
        config['calibrated_on'] = f'Mia calibration.jsonl ({rows} rows, sha256 {digest[:12]})'
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        report = {'model': args.model, 'method': method, 'temperature': temperature, 'rows': rows,
                  'before': trainer.metrics(logits, gold), 'after': trainer.metrics(logits, gold, temperature)}
        print(f"calibration: temperature {temperature:.4f}, ece {report['before']['ece']:.4f} -> {report['after']['ece']:.4f}")

    elif args.throughput:
        import torch
        texts, labels, *_ = load_split(eval_dir / 'unseen-test.jsonl', eval_dir / 'manifest.json')
        s.probs(texts[:512])
        torch.cuda.synchronize()
        started = time.time()
        probs = s.probs(texts)
        torch.cuda.synchronize()
        seconds = time.time() - started
        acc = sum(max(range(3), key=p.__getitem__) == y for p, y in zip(probs, labels)) / len(labels)
        report = {'model': args.model, **s.describe(), 'device': torch.cuda.get_device_name(args.device), 'batch_size': args.batch_size,
                  'rows': len(texts), 'seconds': seconds, 'rows_per_second': len(texts) / seconds, 'holdout_accuracy_check': acc}
        print(f"throughput: {report['rows_per_second']:.1f} rows/s ({len(texts)} rows, {seconds:.1f} s), acc check {acc:.4f}")
    elif args.data:
        texts, labels, digest, rows, manifest = load_split(args.data, args.data.parent / 'manifest.json')
        m = metrics_from_probs(s.probs(texts), labels)
        report = {'model': args.model, 'method': method, **s.describe(), 'requested_device': args.device, 'temperature': [1.0, 1.0, 1.0],
                  'dataset': {'path': str(args.data), 'sha256': digest, 'manifest_rows': rows, 'evaluated_rows': len(texts), 'limit': 0},
                  'metrics': m, 'label_provenance': manifest.get('fields_used_as_text_or_target', {}).get('target', 'dataset gold labels')}
        print(f"{args.data.name}: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} neutral_recall={m['per_class']['中性']['recall']:.3f} ece={m['ece']:.4f}")
    else:
        prep = module('prepare-laya-sentiment.py', 'laya_sentiment_preparation')
        trainer = module('train-qwen-sentiment.py', 'qwen_sentiment_trainer')
        report = {'model': args.model, 'method': method, **s.describe(), 'training_data': None, 'labels': LABELS, 'datasets': {}}
        for path in trainer.CSVS:
            with path.open(newline='', encoding='utf-8-sig') as f:
                rows = [(prep.normalize_text(r['text']), r['label'].strip()) for r in csv.DictReader(f)]
            m = metrics_from_probs(s.probs([t for t, _ in rows]), [LABELS.index(l) for _, l in rows])
            report['datasets'][path.name] = {'path': str(path), 'label_counts': dict(Counter(l for _, l in rows)), 'exact_train_text_overlap': None, 'metrics': m}
            print(f"{path.name}: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} neutral_recall={m['per_class']['中性']['recall']:.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Time a released Mia classifier end to end on the holdout texts, the way evaluate-clm-sentiment.py and
evaluate-systemone-zero-shot.py --throughput time theirs: model loaded beforehand, 512 rows of warm-up, then every
holdout text through tokenisation, forward and softmax.

  python scripts/benchmark-throughput.py --model-type qwen --model models/mia-qwen3.5-2b/final --output reports/mia-qwen3.5-2b/throughput.json
  python scripts/benchmark-throughput.py --model-type laya --model models/mia-laya/final --output reports/mia-laya/throughput.json
  python scripts/benchmark-throughput.py --model-type hf --model IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment --output reports/open-models/throughput-<slug>.json
"""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parent
LABELS = ['负面', '中性', '正面']


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def qwen_runner(model_dir, device, batch_size):
    import torch
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5TextForSequenceClassification
    trainer = module('train-qwen-sentiment.py', 'qwen_sentiment_trainer')
    cfg = json.loads((model_dir / 'sentiment-config.json').read_text(encoding='utf-8'))
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3_5TextForSequenceClassification.from_pretrained(model_dir, dtype=torch.bfloat16).to(device).eval()

    def run(texts):
        ids = trainer.encode(tok, texts, 8192)
        logits, _ = trainer.collect_logits(model, ids, [0] * len(texts), batch_size, tok.pad_token_id, torch.device(device))
        return (logits / float(cfg['temperature'])).softmax(-1).tolist()
    return run, 'transformers Qwen3_5TextForSequenceClassification, bf16, PyTorch fallback kernels for the linear-attention layers'


def laya_runner(model_dir, device, batch_size):
    from laya import load
    agent = load(str(model_dir), device=device)
    questions = json.loads((model_dir.parent / 'questions.json').read_text(encoding='utf-8'))

    def run(texts):
        out = []
        for start in range(0, len(texts), 512):
            for r in agent.predict_batch([{'text': t} for t in texts[start:start + 512]], questions, batch_size=batch_size, sort_by_length=True):
                p = r['answers']['sentiment']['probabilities']
                out.append([p[l] for l in LABELS])
        return out
    return run, 'laya 0.3.20 predict_batch, sort_by_length, chunks of 512 rows'


def hf_runner(model_id, device, batch_size):
    bench = module('evaluate-public-benchmarks.py', 'public_benchmarks')

    def run(texts):
        return bench.predict_hf(str(model_id), texts, device, batch_size)
    return run, 'transformers AutoModelForSequenceClassification, fp32 (library default), texts sorted by length'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-type', choices=('qwen', 'laya', 'hf'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import torch
    rows = [json.loads(line) for line in (ROOT.parent / 'data' / 'eval' / 'unseen-test.jsonl').open(encoding='utf-8')]
    texts = [json.loads(r['state'])['text'] for r in rows]
    labels = [LABELS.index(json.loads(r['gold'])['sentiment']['label']) for r in rows]
    run, path = {'qwen': qwen_runner, 'laya': laya_runner, 'hf': hf_runner}[args.model_type](args.model, args.device, args.batch_size)
    run(texts[:512])
    torch.cuda.synchronize()
    started = time.time()
    probs = run(texts)
    torch.cuda.synchronize()
    seconds = time.time() - started
    acc = sum(max(range(3), key=p.__getitem__) == y for p, y in zip(probs, labels)) / len(labels)
    report = {'model': str(args.model), 'path': path, 'device': torch.cuda.get_device_name(args.device), 'batch_size': args.batch_size,
              'rows': len(texts), 'seconds': seconds, 'rows_per_second': len(texts) / seconds, 'holdout_accuracy_check': acc}
    print(f"throughput: {report['rows_per_second']:.1f} rows/s ({len(texts)} rows, {seconds:.1f} s), acc check {acc:.4f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

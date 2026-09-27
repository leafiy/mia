#!/usr/bin/env python3
"""Lay Mia's data out for CLM's own trainer, `train/finetune.py --task choice`, and pre-fill its embedding cache.

  .venv-laya/bin/python scripts/prepare-clm-sentiment.py --clm-repo /path/to/CLM \
    --embed-model /path/to/Qwen3-8B --train <train.jsonl> --test data/eval/test.jsonl --out <workdir>

  .venv-laya/bin/python /path/to/CLM/train/finetune.py --task choice --data <workdir>/data --workflow sentiment \
    --embed-model /path/to/Qwen3-8B --embed-cache <workdir>/embeddings --init-ckpt CLM_v0.1-8B.pt \
    --out-dir <workdir>/runs/<tag>

The JSONL rows already use the typed-decisions wire format (state / questions / gold as JSON strings), so they are
copied unchanged into <out>/data/sentiment/{train,test}-00000.parquet. Every state and option text finetune.py will
look up is embedded by clm-embed.py (Qwen3-8B last-token pooling on CLM's own token ids) into
<out>/embeddings/choice_<model>_<max_len>.npz, the file finetune.py reads, so it finds nothing missing and never
starts vLLM. finetune.py carves its validation set (--val-frac, default 10%) out of train; `test` is only reported.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--clm-repo', type=Path, required=True)
    parser.add_argument('--embed-model', type=Path, required=True, help='Qwen3-8B checkpoint directory')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT.parent / 'data' / 'eval' / 'manifest.json',
                        help='checks train.jsonl / test.jsonl against the SHA-256 recorded here')
    parser.add_argument('--workflow', default='sentiment')
    parser.add_argument('--max-len', type=int, default=2048, help='finetune.py --max-len (choice default 2048)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    embed = module('clm-embed.py', 'clm_embed')
    clm = embed.clm_import(args.clm_repo)
    finetune, adapters = clm[0], clm[1]

    files = json.loads(args.manifest.read_text(encoding='utf-8'))['files']
    data_dir = args.out / 'data' / args.workflow
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, path in (('train', args.train), ('test', args.test)):
        expected = files[path.name]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError(f'{path}: SHA-256 differs from {args.manifest}')
        rows = [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]
        if len(rows) != expected['rows']:
            raise ValueError(f'{path}: {len(rows)} rows, manifest says {expected["rows"]}')
        table = pa.Table.from_pylist([{'id': r['id'], 'workflow': args.workflow, 'state': r['state'],
                                       'questions': r['questions'], 'gold': r['gold']} for r in rows])
        pq.write_table(table, data_dir / f'{split}-00000.parquet')
        print(f'[data] {split}: {len(rows)} rows -> {data_dir / f"{split}-00000.parquet"}', flush=True)

    texts = []
    for split in ('train', 'test'):
        for e in adapters.typed_decision_examples(finetune.load_typed_rows(str(args.out / 'data'), split, args.workflow, None)):
            texts += [e.state_text, *e.candidates]
    texts = list(dict.fromkeys(texts))
    cache_path = args.out / 'embeddings' / f'choice_{finetune._slug(str(args.embed_model))}_{args.max_len}.npz'
    encoder = embed.CachedEncoder(clm, args.embed_model, cache_path, args.device, args.batch_size, args.max_len)
    print(f'[cache] {len(set(texts))} distinct texts, {len(encoder.cache.missing(texts))} to embed -> {cache_path}', flush=True)
    encoder.ensure(texts)
    print('[done] finetune.py will find every text in the cache', flush=True)


if __name__ == '__main__':
    main()

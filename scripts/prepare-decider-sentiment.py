#!/usr/bin/env python3
"""Build the data cache decider's own trainer (`python -m decider.train`) reads, to continue decider-2b on Mia's data.

  PYTHONPATH=<decider repo> .venv-laya/bin/python scripts/prepare-decider-sentiment.py --decider-repo <decider repo> \
    --train <train.jsonl> --out <workdir>/data/mia-decider.pkl

  PYTHONPATH=<decider repo> .venv-laya/bin/python -m decider.train --model <decider-2b> --data <workdir>/data/mia-decider.pkl \
    --out <workdir>/runs/<tag> --epochs 1 --lr 8e-6 --warmup 50 --max_tokens 16384 --accum 2 --max_options 255 \
    --max_ctx 16384 --none_prob 0.1 --schema_first_prob 0.5 --eval_every 100 --eval_limit 600

Every Mia row becomes the typed question the zero-shot baseline asked (evaluate-systemone-sentiment.py QUESTIONS), rendered
by decider.data.teacher_questions.to_example, i.e. exactly as decider's /v1/systemone path renders a request. To keep the
general decision ability the model was chosen for, a replay sample of decider's own teacher data (teacher_data/, the
non-held-out domains, built by decider.data.mixture.teacher_sets) is mixed in at --replay-frac of the final set. The six
teacher domains decider never trained on (mixture.HELD_DOMAINS) become evaluation sets, so the trainer's periodic eval shows
whether general decisions on unseen domains survive; Mia's calibration split is the in-task eval. Dev and holdout stay
untouched.
"""
import argparse
import hashlib
import importlib.util
import json
import pickle
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mia_examples(path, expected, D, S1, to_example, questions, task):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
        raise ValueError(f'{path}: SHA-256 differs from manifest')
    q = questions['sentiment']
    out = []
    for line in path.open(encoding='utf-8'):
        row = json.loads(line)
        rec = {'state': json.loads(row['state'])['text'],
               'questions': [{**q, 'answer': json.loads(row['gold'])['sentiment']['label']}]}
        out.append(to_example(rec, D, S1, task))
    if len(out) != expected['rows']:
        raise ValueError(f'{path}: {len(out)} rows, manifest says {expected["rows"]}')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--decider-repo', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True, help="Mia's train.jsonl")
    parser.add_argument('--manifest', type=Path, default=ROOT.parent / 'data' / 'eval' / 'manifest.json')
    parser.add_argument('--replay-frac', type=float, default=0.15, help='share of teacher replay rows in the final training set')
    parser.add_argument('--heldout-per-set', type=int, default=600)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    from decider import data as D
    from decider import systemone as S1
    from decider.data import mixture
    from decider.data.teacher_questions import to_example
    questions = module('evaluate-systemone-sentiment.py', 'systemone_sentiment').QUESTIONS
    files = json.loads(args.manifest.read_text(encoding='utf-8'))['files']
    rng = random.Random(args.seed)

    train = mia_examples(args.train, files[args.train.name], D, S1, to_example, questions, 'mia_sentiment')
    cal_path = args.manifest.parent / 'calibration.jsonl'
    evals = {'mia_calibration': mia_examples(cal_path, files[cal_path.name], D, S1, to_example, questions, 'mia_calibration')}

    mixture.TEACHER = str(args.decider_repo / 'teacher_data')
    recs, routes, commands = mixture.load_teacher()
    pool = [e for exs in mixture.teacher_sets(recs, routes, rng, commands).values() for e in exs]
    n_replay = min(len(pool), round(len(train) * args.replay_frac / (1 - args.replay_frac)))
    replay = rng.sample(pool, n_replay)
    held = mixture.HELD_DOMAINS
    custom_held = [to_example(r, D, S1, 'teacher_custom_heldout') for r in recs if r['domain'] in held]
    routing_held = [to_example(r, D, S1, 'teacher_routing_heldout') for r in routes if r['domain'] in held]
    rng.shuffle(custom_held); rng.shuffle(routing_held)
    evals['teacher_custom_heldout'] = custom_held[:args.heldout_per_set]
    evals['teacher_routing_heldout'] = routing_held[:args.heldout_per_set]

    mixed = train + replay
    rng.shuffle(mixed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('wb') as f:
        pickle.dump((mixed, evals), f)
    summary = {'train_rows': len(mixed), 'mia_rows': len(train), 'replay_rows': len(replay), 'replay_pool': len(pool),
               'replay_tasks': dict(Counter(e.task for e in replay)), 'held_domains': sorted(held),
               'evals': {k: len(v) for k, v in evals.items()}, 'questions': questions,
               'example_question': {'text': train[0].qs[0].text, 'options': train[0].qs[0].options}}
    args.out.with_suffix('.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

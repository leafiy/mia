#!/usr/bin/env python3
"""Qwen3-8B last-token embeddings for CLM heads, computed with transformers instead of vLLM.

CLM (https://github.com/Contrastive-LM/CLM) scores a (state, option) pair as
exp(logit_scale) * cos(state_head(s), action_head(a)), where s and a are Qwen3-8B last-token hidden states,
L2-normalised: what `vllm serve Qwen/Qwen3-8B --runner pooling` returns. This module computes the same vectors
from the same token ids (CLM's embed_utils.Recipe) with transformers, so no vLLM is needed.

  .venv-laya/bin/python scripts/clm-embed.py --clm-repo /path/to/CLM --embed-model /path/to/Qwen3-8B \
    --head /path/to/CLM_v0.1-8B.pt --self-check

--self-check replays the two README examples of the CLM repo (tides ranking, support-ticket questions) with the
reference head; it should print prob ~0.997 for the Moon and department=billing.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def clm_import(repo):
    """Put a CLM checkout on sys.path and return (finetune, adapters, embed_utils, schema, heads)."""
    repo = Path(repo).resolve()
    for sub in ('src', 'preprocessing', 'train'):
        path = str(repo / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    import adapters
    import embed_utils
    import finetune
    from clm import heads, schema
    return finetune, adapters, embed_utils, schema, heads


class LastTokenEmbedder:
    """Final-norm hidden state of the last token, L2-normalised; right padding, so padding never reaches it."""

    def __init__(self, model_dir, device='cuda:0', batch_size=64, max_batch_tokens=16384):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16).to(self.device).eval()
        self.hidden = self.model.config.hidden_size
        self.batch_size, self.max_batch_tokens = batch_size, max_batch_tokens

    def embed(self, id_lists, log_every=0):
        torch = self.torch
        out = np.zeros((len(id_lists), self.hidden), dtype=np.float32)
        order = sorted(range(len(id_lists)), key=lambda i: len(id_lists[i]))
        start, done, started = 0, 0, time.time()
        with torch.inference_mode():
            while start < len(order):
                stop = start + 1
                while (stop < len(order) and stop - start < self.batch_size
                       and (stop - start + 1) * len(id_lists[order[stop]]) <= self.max_batch_tokens):
                    stop += 1
                idx = order[start:stop]
                width = len(id_lists[idx[-1]])
                ids = torch.zeros((len(idx), width), dtype=torch.long)
                mask = torch.zeros((len(idx), width), dtype=torch.long)
                for row, i in enumerate(idx):
                    seq = id_lists[i]
                    ids[row, :len(seq)] = torch.tensor(seq)
                    mask[row, :len(seq)] = 1
                hidden = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                    use_cache=False).last_hidden_state
                last = (mask.sum(1) - 1).to(self.device)
                vec = hidden[torch.arange(len(idx), device=self.device), last].float()
                out[idx] = torch.nn.functional.normalize(vec, dim=-1).cpu().numpy()
                before, done, start = done, done + len(idx), stop
                if log_every and done // log_every != before // log_every:
                    print(f'[embed] {done}/{len(order)} {done / (time.time() - started):.1f} texts/s', flush=True)
        return out


class CachedEncoder:
    """Texts -> embeddings through CLM's own TextCache (sha1(text) -> float16 vector, .npz)."""

    def __init__(self, clm, embed_model, cache_path, device, batch_size, max_len=2048):
        finetune, _, embed_utils, _, _ = clm
        self.cache = finetune.TextCache(str(cache_path))
        self.recipe = embed_utils.Recipe(str(embed_model), max_len)
        self.embed_model, self.device, self.batch_size = embed_model, device, batch_size
        self.embedder = None

    def ensure(self, texts, save_every=20000, log_every=2000):
        """Embed and persist the texts the cache does not hold yet."""
        todo = self.cache.missing(texts)
        if todo and self.embedder is None:
            self.embedder = LastTokenEmbedder(self.embed_model, self.device, self.batch_size)
        for start in range(0, len(todo), save_every):
            part = todo[start:start + save_every]
            ids = [self.recipe.text_ids(t, keep='tail') for t in part]
            self.cache.add(part, self.embedder.embed(ids, log_every=log_every))
            print(f'[cache] {min(start + save_every, len(todo))}/{len(todo)} new texts saved', flush=True)

    def vectors(self, texts):
        self.ensure(texts)
        return np.stack([self.cache[t] for t in texts]).astype(np.float32)


def choice_logits(pair, state_vecs, option_vecs):
    """[n, k] logits exp(logit_scale) * cos for a clm.heads.HeadPair; option_vecs is [k, hidden] shared by all rows."""
    zs, zc = pair.project(state_vecs.astype(np.float32), option_vecs.astype(np.float32))
    return pair.scale * zs @ zc.T


def self_check(args):
    clm = clm_import(args.clm_repo)
    _, _, embed_utils, schema, heads = clm
    recipe = embed_utils.Recipe(str(args.embed_model), 2048)
    embedder = LastTokenEmbedder(args.embed_model, args.device)
    pair = heads.HeadPair('head', str(args.head), device=args.device).ensure()

    def answer(state, questions):
        pairs = schema.build_pairs(state, questions)
        states = [p[0] for p in pairs.values()]
        cands = [t for p in pairs.values() for t in p[2]]
        vecs = embedder.embed([recipe.text_ids(t, keep='tail') for t in states + cands])
        zs, zc = pair.project(vecs[:len(states)], vecs[len(states):])
        out, k = {}, 0
        for i, (qid, (_, keys, texts)) in enumerate(pairs.items()):
            out[qid] = schema.answer_from_logits(questions[qid], keys, (pair.scale * zc[k:k + len(texts)] @ zs[i]).tolist())
            k += len(texts)
        return out

    tides = answer('What causes tides on Earth?', {'rank': {'type': 'choice', 'instructions': None, 'criteria': {
        '0': "The Moon's gravitational pull.", '1': 'Photosynthesis in plants.'}}})['rank']
    print('tides:', tides['probabilities'], '(README: 0.997 for the Moon)')
    ticket = answer('Customer: my invoice was charged twice and nobody answers!', {
        'urgency': {'type': 'noul', 'instructions': 'Is this urgent?'},
        'department': {'type': 'choice', 'instructions': 'Which team should handle this?',
                       'criteria': {'billing': 'Charges, invoices, refunds', 'technical': 'Bugs and outages'}},
        'frustration': {'type': 'score', 'instructions': 'How frustrated is the customer?',
                        'criteria': ['Calm', 'Frustrated', 'Very angry']}})
    print('ticket department:', ticket['department']['choice'], ticket['department']['probabilities'], '(README: billing)')
    print('ticket urgency:', ticket['urgency'], 'frustration:', ticket['frustration']['score'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--clm-repo', type=Path, required=True)
    parser.add_argument('--embed-model', type=Path, required=True, help='Qwen3-8B checkpoint directory')
    parser.add_argument('--head', type=Path, required=True, help='CLM head checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--self-check', action='store_true', required=True)
    self_check(parser.parse_args())


if __name__ == '__main__':
    main()

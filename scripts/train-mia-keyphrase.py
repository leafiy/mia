#!/usr/bin/env python3
"""Train mia-keyphrase-2b: Qwen3.5-2B + low-rank adapters, merged into the weights at the end.

  python scripts/train-mia-keyphrase.py --base /path/to/Qwen3.5-2B --train train.jsonl --output models/mia-keyphrase-2b/final

train.jsonl: one JSON object per line with "text" (a passage of up to 500 characters) and "keyphrases"
(phrases copied verbatim from the passage, most important first). Phrases not found in the text are dropped.
The first 200 rows after shuffling are held out as a dev set. The training set of the released model is not distributed.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

USER = "摘出下面这段材料的关键短语，按重要性从高到低排列，只输出 JSON。\n材料：{text}"


def target(row):
    phrases = list(dict.fromkeys(k for k in row["keyphrases"] if k and k in row["text"]))
    return json.dumps({"keyphrases": phrases}, ensure_ascii=False, separators=(",", ":"))


class LoRA(torch.nn.Module):
    """Frozen bf16 base layer plus a trainable low-rank update."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features, device="cuda") / math.sqrt(base.in_features))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device="cuda"))

    def forward(self, x):
        low = torch.nn.functional.linear(torch.nn.functional.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))
        return self.base(x) + low * self.scale

    def merge(self):
        self.base.weight.data += (self.B @ self.A * self.scale).to(self.base.weight.dtype)
        return self.base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    args = ap.parse_args()

    data = []
    for line in open(args.train, encoding="utf-8"):
        row = json.loads(line)
        data.append((row["text"], target(row)))
    random.Random(7).shuffle(data)
    dev, train = data[:200], data[200:]
    print("train", len(train), "dev", len(dev), flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "right"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.base, dtype=torch.bfloat16).cuda()
    for p in model.parameters():
        p.requires_grad = False
    lm = model.model.language_model
    targets = [(n, m) for n, m in lm.named_modules() if isinstance(m, torch.nn.Linear) and "embed" not in n]
    for name, module in targets:
        parent, leaf = name.rsplit(".", 1)
        setattr(lm.get_submodule(parent), leaf, LoRA(module, args.rank, args.alpha))
    print("adapted layers", len(targets), flush=True)

    def prompt(text):
        return tok.apply_chat_template([{"role": "user", "content": USER.format(text=text)}],
                                       add_generation_prompt=True, enable_thinking=False, tokenize=False)

    def loss_of(rows):
        ids, labels = [], []
        for text, y in rows:
            a = tok(prompt(text), add_special_tokens=False)["input_ids"][-1000:]
            b = tok(y + "<|im_end|>", add_special_tokens=False)["input_ids"]
            ids.append(a + b)
            labels.append([-100] * len(a) + b)
        n = max(map(len, ids))
        x = torch.tensor([i + [tok.pad_token_id] * (n - len(i)) for i in ids]).cuda()
        y = torch.tensor([l + [-100] * (n - len(l)) for l in labels]).cuda()
        mask = (x != tok.pad_token_id) | (y != -100)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(input_ids=x, attention_mask=mask).last_hidden_state[:, :-1]
            keep = y[:, 1:] != -100   # the vocabulary has 248k entries: run lm_head on target positions only
            return torch.nn.functional.cross_entropy(model.lm_head(hidden[keep]).float(), y[:, 1:][keep])

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    warm, step, started = max(10, steps // 20), 0, time.time()
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(train)
        train.sort(key=lambda r: len(r[0]) // 60 + random.random())   # rough length buckets
        batches = [train[i:i + args.batch_size] for i in range(0, len(train), args.batch_size)]
        random.Random(100 + epoch).shuffle(batches)
        for batch in batches:
            lr = args.lr * (step + 1) / warm if step < warm else args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for group in opt.param_groups:
                group["lr"] = lr
            loss = loss_of(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f} t={time.time() - started:.0f}s", flush=True)
        model.eval()
        with torch.no_grad():
            dev_loss = sum(loss_of(dev[i:i + 16]).item() * len(dev[i:i + 16]) for i in range(0, len(dev), 16)) / len(dev)
        model.train()
        print(f"epoch {epoch + 1} dev_loss {dev_loss:.4f}", flush=True)

    for name, _ in targets:
        parent, leaf = name.rsplit(".", 1)
        setattr(lm.get_submodule(parent), leaf, getattr(lm.get_submodule(parent), leaf).merge())
    model.save_pretrained(Path(args.output), safe_serialization=True)
    tok.save_pretrained(Path(args.output))
    print("SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()

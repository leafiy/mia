---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
language: zh
---
# mia-evidence-2b

Mia 系列的依据词模型：给一条中文短文本和它的整体态度（积极 / 消极 / 中性，由 mia-decider-2b 判出），输出这句话里承载态度的词和表达模式。

- 依据词：最多 5 个，逐字摘自原文，各带对本句的贡献方向、28 类词语标签之一、固有极性。
- 表达模式：12 类之一（正向直述、否定翻转、反讽反转、程度强化、情感对比、中性陈述、让步转折、反问评价、条件催促、话语引入、惯用语整体、客观状态陈述），另有是否真值否定和最关键的触发短语。

底座 Qwen/Qwen3.5-2B，低秩适配训练后并回权重，bf16。输入模板、标签表在 `final/evidence-config.json`（本目录另放了一份），成绩在 `mia.json` 和 `reports/mia-evidence-2b/`，训练过程见 [docs/mia-series.md](../../docs/mia-series.md)。权重从 [ModelScope leafiy2/mia-evidence-2b](https://modelscope.cn/models/leafiy2/mia-evidence-2b) 下载。

400 条留出文本上与标注模型对比：摘出的词 98.9% 逐字在原文里，摘词查准 0.806、查全 0.795，表达模式一致率 0.677（整体态度取标注模型自己的判断时 0.784；标注模型自己标两遍 0.790）。

用法：

```python
import json, torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
tok = AutoTokenizer.from_pretrained("final"); tok.padding_side = "left"
model = Qwen3_5ForConditionalGeneration.from_pretrained("final", dtype=torch.bfloat16).cuda().eval()
cfg = json.load(open("final/evidence-config.json", encoding="utf-8"))
msg = [{"role": "user", "content": cfg["user_template"].format(text="这家店不错，就是上菜慢", sentiment="中性")}]
enc = tok([tok.apply_chat_template(msg, add_generation_prompt=True, enable_thinking=False, tokenize=False)], return_tensors="pt", add_special_tokens=False).to("cuda")
out = model.generate(**enc, max_new_tokens=512, do_sample=False, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

项目主页：https://github.com/leafiy/mia

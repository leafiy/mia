---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
language: zh
---
# mia-keyphrase-2b

Mia 系列的关键短语模型：给一段不超过 500 字的中文材料，摘出最能代表它在讲什么的关键短语，最多 10 个，逐字来自原文，按重要性从高到低排列。

底座 Qwen/Qwen3.5-2B，低秩适配训练后并回权重，bf16。输入模板在 `final/keyphrase-config.json`（本目录另放了一份），成绩在 `mia.json` 和 `reports/mia-keyphrase-2b/`，训练过程见 [docs/mia-series.md](../../docs/mia-series.md)。权重从 [ModelScope leafiy2/mia-keyphrase-2b](https://modelscope.cn/models/leafiy2/mia-keyphrase-2b) 下载。

288 段留出材料上逐个短语盲评：合格 58.0%，勉强 39.2%，不合格 2.7%，参考答案里的核心短语有 80.0% 被摘到。同一批材料上，StructBERT 关键短语模型（会议转写数据训练）合格 17.3%、不合格 50.6%、核心短语摘到 47.5%。

用法：

```python
import json, torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
tok = AutoTokenizer.from_pretrained("final"); tok.padding_side = "left"
model = Qwen3_5ForConditionalGeneration.from_pretrained("final", dtype=torch.bfloat16).cuda().eval()
cfg = json.load(open("final/keyphrase-config.json", encoding="utf-8"))
msg = [{"role": "user", "content": cfg["user_template"].format(text="城市更新行动进入第三年，老旧小区改造的重点从外立面翻新转向管网更新和加装电梯。")}]
enc = tok([tok.apply_chat_template(msg, add_generation_prompt=True, enable_thinking=False, tokenize=False)], return_tensors="pt", add_special_tokens=False).to("cuda")
out = model.generate(**enc, max_new_tokens=cfg["max_new_tokens"], do_sample=False, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

输出是紧凑写法的 JSON：`{"keyphrases":["...","..."]}`。短语里偶尔带没转义的引号，去掉首尾后按 `","` 切开比 `json.loads` 稳。

项目主页：https://github.com/leafiy/mia

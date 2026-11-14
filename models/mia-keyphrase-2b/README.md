---
license: apache-2.0
language:
- zh
pipeline_tag: text-generation
library_name: transformers
base_model: Qwen/Qwen3.5-2B
tags:
- keyphrase-extraction
- keyword-extraction
- chinese-keyword-extraction
- chinese
- 关键词提取
- 关键词抽取
- 关键短语抽取
- 中文关键词提取
- mia
tasks:
- text-generation
domain:
- nlp
frameworks:
- pytorch
---

# Mia 中文关键词提取 · mia-keyphrase-2b

给一段不超过 500 字的中文材料，摘出最能代表它在讲什么的关键短语，最多 10 个，逐字来自原文，按重要性从高到低排列。摘出来的是完整的说法，很少切碎。文章、报告和拼在一起的多条评论都能用。

Chinese keyphrase extraction model: up to 10 key phrases copied verbatim from the text, ranked by importance.

[GitHub](https://github.com/leafiy/mia) · [ModelScope](https://modelscope.cn/models/leafiy2/mia-keyphrase-2b)

## 效果

300 段没进训练的材料，其中 288 段有明确主题。把四家摘的短语混在一起打乱，由一个没参与标注的大模型逐个盲评（合格 / 勉强 / 不合格）。它在看到任何输出之前，先自己写下了每段的核心短语：

| | 合格 | 勉强 | 不合格 | 核心短语被摘到 |
| --- | ---: | ---: | ---: | ---: |
| `mia-keyphrase-2b` | 58.0% | 39.2% | 2.7% | 80.0% |
| 同一配方换 Qwen3.5-4B | 59.1% | 38.5% | 2.4% | 80.8% |
| 标注用的大模型 | 57.2% | 38.1% | 4.7% | 80.5% |
| StructBERT 关键短语模型（会议转写数据训练） | 17.3% | 32.1% | 50.6% | 47.5% |

- **追平了教它的大模型。** 合格率和核心短语覆盖都和标注用的大模型持平，不合格的更少。
- **StructBERT 关键短语模型一半不合格。** 它在会议转写数据上训练，拿到政策、评论、论文这类材料，摘出来的多是被切开的泛词。
- **词没有被切碎。** 摘出的短语两端落在分词边界上的占 98%，切在词中间的只占全部短语的 1%。

## 快速开始

```bash
pip install "torch>=2.14" "transformers>=5.17" modelscope
modelscope download --model leafiy2/mia-keyphrase-2b --local_dir mia-keyphrase-2b
(cd mia-keyphrase-2b && sha256sum -c final.sha256)
```

```python
import json, torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

final = 'mia-keyphrase-2b/final'
tok = AutoTokenizer.from_pretrained(final); tok.padding_side = 'left'
model = Qwen3_5ForConditionalGeneration.from_pretrained(final, dtype=torch.bfloat16).cuda().eval()
cfg = json.load(open(final + '/keyphrase-config.json', encoding='utf-8'))

text = '城市更新行动进入第三年，老旧小区改造的重点从外立面翻新转向管网更新和加装电梯。'   # 不超过 500 字，长材料先切段
msg = [{'role': 'user', 'content': cfg['user_template'].format(text=text)}]
enc = tok([tok.apply_chat_template(msg, add_generation_prompt=True, enable_thinking=False, tokenize=False)],
          return_tensors='pt', add_special_tokens=False).to('cuda')
out = model.generate(**enc, max_new_tokens=cfg['max_new_tokens'], do_sample=False, eos_token_id=tok.convert_tokens_to_ids('<|im_end|>'))
raw = tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
print(raw[len('{"keyphrases":["'):-len('"]}')].split('","'))   # 短语里偶尔带没转义的引号，这样切比 json.loads 稳
```

输出是紧凑写法的 JSON：`{"keyphrases":["...","..."]}`。

## 用之前要知道

- 一次只看 500 字，判断不了一个短语在整篇里重不重要，长文章要先切段。
- 近四成短语被判为勉强，多是偏泛或属于次要细节。

## 模型信息

| | |
| --- | --- |
| 底座 | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)，生成模型 |
| 输入 | 一段不超过 500 字的材料，模板在 `final/keyphrase-config.json`（本目录另放了一份） |
| 训练数据 | 4,000 段中文材料，每个来源文件最多取 3 段，训练和测试来自不同文件；由一个大模型按统一规则摘 5 到 10 个逐字出自原文的短语并排序 |
| 训练 | 低秩适配（r=32），lr 1e-4，batch 8，3 轮，RTX 4090 上 439 秒，适配已并回权重 |
| 权重 | bf16，约 4.4G |
| 速度 | RTX 4090 约 17 段/秒 |
| 依赖 | `torch>=2.14`，`transformers>=5.17` |
| 许可 | Apache-2.0 |

完整报告和训练脚本在 [GitHub](https://github.com/leafiy/mia) 的 `reports/mia-keyphrase-2b/` 和 `scripts/train-mia-keyphrase.py`。评测为什么改成逐个盲评、为什么换 4B 底座没用，见 [docs/mia-series.md](https://github.com/leafiy/mia/blob/main/docs/mia-series.md)。

## Mia 系列

| 模型 | 做什么 | 下载 |
| --- | --- | --- |
| `mia-decider-2b`（推荐） | 情感分析：正面 / 负面 / 中性 | [Hugging Face](https://huggingface.co/leafiy/mia-decider-2b) · [ModelScope](https://modelscope.cn/models/leafiy2/mia-decider-2b) |
| `mia-qwen3.5-2b` | 情感分析，只用 transformers 就能加载 | [Hugging Face](https://huggingface.co/leafiy/mia-qwen3.5-2b) · [ModelScope](https://modelscope.cn/models/leafiy2/mia-qwen3.5-2b) |
| `mia-laya` | 情感分析轻量版，每秒近 2,000 条 | [Hugging Face](https://huggingface.co/leafiy/mia-laya) · [ModelScope](https://modelscope.cn/models/leafiy2/mia-laya) |
| `mia-vad-2b` | 多维情绪识别：效价、唤醒度、支配度、七类情绪、强度 | [ModelScope](https://modelscope.cn/models/leafiy2/mia-vad-2b) |
| `mia-evidence-2b` | 情感依据词提取 | [ModelScope](https://modelscope.cn/models/leafiy2/mia-evidence-2b) |
| `mia-keyphrase-2b` | 中文关键词提取 | [ModelScope](https://modelscope.cn/models/leafiy2/mia-keyphrase-2b) |

## 引用

```bibtex
@misc{mia2026,
  title  = {Mia: Chinese Sentiment Analysis Models},
  author = {leafiy},
  year   = {2026},
  url    = {https://github.com/leafiy/mia}
}
```

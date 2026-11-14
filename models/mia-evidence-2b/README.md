---
license: apache-2.0
language:
- zh
pipeline_tag: text-generation
library_name: transformers
base_model: Qwen/Qwen3.5-2B
tags:
- sentiment-analysis
- explainable-sentiment-analysis
- opinion-mining
- opinion-extraction
- sentiment-evidence-extraction
- chinese
- 情感分析
- 可解释情感分析
- 情感词提取
- 观点抽取
- mia
tasks:
- text-generation
domain:
- nlp
frameworks:
- pytorch
---

# Mia 中文情感依据词提取 · mia-evidence-2b

情感分析只告诉你一条评论是正面还是负面，这个模型告诉你凭什么。给它一条中文短文本和整体态度，它从原文里摘出决定态度的词或短语，并判断整句是怎么表达的：直说、否定翻转、反讽，还是让步转折。

Explainable Chinese sentiment analysis: extracts the words that carry the sentiment, and labels how the sentence expresses it (negation flip, irony, concession, ...).

[GitHub](https://github.com/leafiy/mia) · [ModelScope](https://modelscope.cn/models/leafiy2/mia-evidence-2b)

## 能给出什么

- **依据词**：最多 5 个，逐字摘自原文，每个都标出在这句里往积极还是消极拉、属于 28 类词语标签里的哪一类（程度副词、否定副词、褒义名词、反讽标记词等），以及词本身的极性。
- **表达模式**：12 类之一，正向直述、否定翻转、反讽反转、程度强化、情感对比、中性陈述、让步转折、反问评价、条件催促、话语引入、惯用语整体、客观状态陈述。
- **否定和触发短语**：这句里有没有真正改变评价方向的否定，最关键的触发短语是哪一段。
- **顺带的标记**：依据词里积极和消极两个方向都有，就说明好评差评并存。

## 效果

400 条没进训练的文本，以标注模型的结果为准。对照是一个通用的 9B 模型用提示词直接作答，以及标注模型把前 200 条再标一遍：

| | 通用 9B 模型 | `mia-evidence-2b` | 标注模型标第二遍 |
| --- | ---: | ---: | ---: |
| 摘出的词逐字在原文里 | 0.960 | **0.989** | 0.987 |
| 摘词查准 | 0.638 | **0.806** | 0.889 |
| 摘词查全 | **0.828** | 0.795 | 0.815 |
| 词语标签判对 | 0.446 | **0.706** | 0.775 |
| 表达模式判对 | 0.463 | **0.677** | 0.790 |

查准查全按字符有重叠算。9B 的做法是多摘，所以漏得少，但三个词里有一个不是依据。整体态度换成标注模型自己判的，表达模式从 0.677 升到 0.784。400 条里标注模型认为好评差评并存的 30 条，从依据词推出 21 条，其中 16 条对。

## 快速开始

```bash
pip install "torch>=2.14" "transformers>=5.17" modelscope
modelscope download --model leafiy2/mia-evidence-2b --local_dir mia-evidence-2b
(cd mia-evidence-2b && sha256sum -c final.sha256)
```

```python
import json, torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

final = 'mia-evidence-2b/final'
tok = AutoTokenizer.from_pretrained(final); tok.padding_side = 'left'
model = Qwen3_5ForConditionalGeneration.from_pretrained(final, dtype=torch.bfloat16).cuda().eval()
cfg = json.load(open(final + '/evidence-config.json', encoding='utf-8'))

text, sentiment = '这家店不错，就是上菜慢', '中性'   # sentiment 用 mia-decider-2b 判出的结果：正面→积极，负面→消极，中性→中性
msg = [{'role': 'user', 'content': cfg['user_template'].format(text=text, sentiment=sentiment)}]
enc = tok([tok.apply_chat_template(msg, add_generation_prompt=True, enable_thinking=False, tokenize=False)],
          return_tensors='pt', add_special_tokens=False).to('cuda')
out = model.generate(**enc, max_new_tokens=cfg['max_new_tokens'], do_sample=False, eos_token_id=tok.convert_tokens_to_ids('<|im_end|>'))
print(tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True))
```

- 整体态度要先由 [mia-decider-2b](https://modelscope.cn/models/leafiy2/mia-decider-2b) 判出来再传进来。
- 输出是紧凑写法的 JSON。依据词里偶尔带没转义的引号，按固定的键顺序逐段读比 `json.loads` 稳。

## 用之前要知道

- 只用通用场景的 3,800 条训过，没有分行业。
- 「是否有否定」这一项各家都不稳，标注模型自己标两遍的 F1 也只有 0.57。

## 模型信息

| | |
| --- | --- |
| 底座 | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)，生成模型 |
| 输入 | 文本 + 整体态度（积极 / 消极 / 中性），模板和标签表在 `final/evidence-config.json`（本目录另放了一份） |
| 训练数据 | 从 Mia 训练集里抽 4,000 条，由一个大模型按统一规则标成 JSON |
| 训练 | 低秩适配（r=32，文本主干的 186 个线性层），lr 1e-4，3 轮，RTX 4090 上 219 秒，适配已并回权重；全量微调成绩与它打平 |
| 权重 | bf16，约 4.4G |
| 速度 | RTX 4090 约 15 条/秒 |
| 依赖 | `torch>=2.14`，`transformers>=5.17` |
| 许可 | Apache-2.0 |

完整报告和训练脚本在 [GitHub](https://github.com/leafiy/mia) 的 `reports/mia-evidence-2b/` 和 `scripts/train-mia-evidence.py`，训练过程见 [docs/mia-series.md](https://github.com/leafiy/mia/blob/main/docs/mia-series.md)。

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

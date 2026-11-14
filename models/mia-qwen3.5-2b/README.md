---
license: apache-2.0
language:
- zh
pipeline_tag: text-classification
library_name: transformers
base_model: Qwen/Qwen3.5-2B
tags:
- sentiment-analysis
- sentiment-classification
- chinese-sentiment-analysis
- text-classification
- chinese
- qwen3.5
- 情感分析
- 中文情感分析
- 情感分类
- 文本分类
- 舆情分析
- mia
tasks:
- text-classification
- sentiment-classification
domain:
- nlp
frameworks:
- pytorch
metrics:
- accuracy
- f1
---

# Mia 中文情感分析 · mia-qwen3.5-2b

判断一段中文是正面、负面还是中性，并给出三类概率。Qwen3.5-2B 加一个分类头全量微调，只用 transformers 就能加载，不用装别的包。从商品评论、微博帖子到问卷开放题和客服对话，都能直接用。

Chinese sentiment analysis model (positive / negative / neutral) fine-tuned from Qwen3.5-2B, loadable with plain `transformers`.

[GitHub](https://github.com/leafiy/mia) · [Hugging Face](https://huggingface.co/leafiy/mia-qwen3.5-2b) · [ModelScope](https://modelscope.cn/models/leafiy2/mia-qwen3.5-2b)

## 亮点

- **中性认得最准。** 客观陈述、转述、好评差评并存的文本都能认出来，中性召回在留出测试上 0.892，在 30 行业 900 条上 0.903，是 Mia 三个情感版本里最高的。
- **准。** 5,372 条留出测试上三分类准确率 87.1%，同样能分中性的开源模型最高 69.3%；30 行业 900 条上 94.1%。
- **比 decider 版快。** RTX 4090 上每秒 626 条，batch 64 只占约 6.8G 显存。

要最准、概率最可信，选 [mia-decider-2b](https://huggingface.co/leafiy/mia-decider-2b)；要最快，选 [mia-laya](https://huggingface.co/leafiy/mia-laya)。

## 和开源中文情感模型对比

同一套测试、同一张 RTX 4090。Mia 没有在下面任何一个公开数据集上训练过。

| 模型 | 类别 | 三分类<br>留出测试 5,372 条 | 三分类<br>30 行业 900 条 | 正负二选一<br>留出测试 | 微博 | 豆瓣影评<br>Macro-F1 | ChnSentiCorp | 网购评论 | eprstmt | 4090<br>每秒条数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [**mia-decider-2b**](https://huggingface.co/leafiy/mia-decider-2b) | 负 / 中 / 正 | **0.881** | 0.940 | **0.978** | **0.830** | **0.519** | 0.880 | 0.921 | 0.893 | 375 |
| [**mia-qwen3.5-2b**](https://huggingface.co/leafiy/mia-qwen3.5-2b) | 负 / 中 / 正 | 0.871 | **0.941** | 0.972 | 0.798 | 0.499 | 0.882 | 0.922 | 0.897 | 626 |
| [**mia-laya**](https://huggingface.co/leafiy/mia-laya) | 负 / 中 / 正 | 0.817 | 0.938 | 0.954 | 0.803 | 0.467 | 0.883 | 0.919 | 0.882 | 1,977 |
| [Erlangshen-Roberta-330M-Sentiment](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment) | 负 / 正 | 0.452 | 0.656 | 0.880 | 0.729 | 0.415 | **0.968** | **0.989** | **0.997** | 1,378 |
| [Erlangshen-Roberta-110M-Sentiment](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment) | 负 / 正 | 0.448 | 0.646 | 0.871 | 0.721 | 0.409 | 0.962 | 0.979 | 0.982 | 3,809 |
| [uer/roberta-base-finetuned-dianping-chinese](https://huggingface.co/uer/roberta-base-finetuned-dianping-chinese) | 负 / 正 | 0.455 | 0.657 | 0.885 | 0.726 | 0.414 | 0.877 | 0.894 | 0.795 | 3,812 |
| [uer/roberta-base-finetuned-jd-binary-chinese](https://huggingface.co/uer/roberta-base-finetuned-jd-binary-chinese) | 负 / 正 | 0.431 | 0.654 | 0.839 | 0.704 | 0.455 | 0.834 | 0.928 | 0.902 | 3,795 |
| [cardiffnlp/twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) | 负 / 中 / 正 | 0.693 | 0.664 | 0.921 | 0.813 | 0.397 | 0.809 | 0.894 | 0.848 | 3,650 |
| [lxyuan/distilbert-base-multilingual-cased-sentiments-student](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student) | 负 / 中 / 正 | 0.486 | 0.560 | 0.885 | 0.797 | 0.447 | 0.852 | 0.903 | 0.864 | 6,699 |
| [tabularisai/multilingual-sentiment-analysis](https://huggingface.co/tabularisai/multilingual-sentiment-analysis) | 五档合为三类 | 0.622 | 0.777 | 0.837 | 0.668 | 0.374 | 0.822 | 0.868 | 0.841 | 6,556 |

三分类是负面、中性、正面三选一，只分正负的模型判不出中性。正负二选一和后面四个公开集只比较正、负两类的概率。商品和酒店评论的三个集子上 Erlangshen 更高，其中 ChnSentiCorp 平均 100 多字，Mia 的训练文本都在 100 字以内。测法和逐项报告见 [GitHub](https://github.com/leafiy/mia/blob/main/docs/report.md)。

## 快速开始

```bash
pip install "torch>=2.14" "transformers>=5.17"
hf download leafiy/mia-qwen3.5-2b --local-dir mia-qwen3.5-2b
# 或 modelscope download --model leafiy2/mia-qwen3.5-2b --local_dir mia-qwen3.5-2b
(cd mia-qwen3.5-2b && sha256sum -c final.sha256)
```

```python
import json, torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import Qwen3_5TextForSequenceClassification

final = 'mia-qwen3.5-2b/final'
cfg = json.load(open(f'{final}/sentiment-config.json', encoding='utf-8'))
tok = AutoTokenizer.from_pretrained(final)          # 右侧 padding，线性注意力层要求如此
model = Qwen3_5TextForSequenceClassification.from_pretrained(final, dtype=torch.bfloat16).cuda().eval()

texts = ['这家店的服务真的没话说', '本次记录涉及这家快递网点，相关信息来自页面与服务记录']
enc = tok([cfg['template'].format(text=t) for t in texts], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
with torch.inference_mode():
    probs = (model(**enc).logits.float() / cfg['temperature']).softmax(-1)
for t, p in zip(texts, probs):
    print(t, dict(zip(cfg['labels'], [round(x, 3) for x in p.tolist()])))
```

输入先做和训练时一样的归一化，用 [`normalize_text`](https://github.com/leafiy/mia/blob/main/scripts/prepare-laya-sentiment.py)。

## 用之前要知道

- 一条里好评差评都有时判为中性。只需要正负两类时，比较正面和负面的概率，别直接取三类里最大的那个。
- 训练文本都在 100 字以内，长评论上准确率会下降。
- 负面召回 0.774 是三类里最低的，漏掉的多是不带评价词的负面事实陈述。

## 模型信息

| | |
| --- | --- |
| 底座 | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) 的文本主干（24 层，hidden 2048，线性注意力 + 每 4 层一层全注意力），去掉视觉塔、`lm_head` 与 MTP 头，接线性分类头 |
| 参数量 | 1,882M |
| 输入模板 | `文本：{text}\n情感倾向：`，右侧 padding，不加特殊 token，取最后一个非 pad token 的隐状态 |
| 输出 | `负面 / 中性 / 正面`（`sentiment-config.json` 的 `labels`）；logits 先除以温度 1.7063 再 softmax |
| 权重 | `final/model.safetensors`，bf16，约 3.6G；`final/config.json` 是 `Qwen3_5TextConfig`，`num_labels=3` |
| 训练数据 | 76,112 条中文短文本，褒贬并存的归为中性 |
| 训练 | 全量微调 2 轮，有效 batch 256，AdamW 主干 2e-5 / 头 1e-4，6% warmup 余弦，RTX 4090 约 40 分钟 |
| 留出测试 5,372 条 | 准确率 0.8710 · Macro-F1 0.8564 · 召回 负 0.774 / 中 0.892 / 正 0.889 |
| 30 行业 900 条 | 准确率 0.9411 · Macro-F1 0.9411 · 中性召回 0.903 |
| 对照 | 未微调的原版 Qwen3.5-2B 用同样的三类定义零样本作答：留出测试 0.6372，900 条 0.7944 |
| 速度 | RTX 4090，bf16，batch 64：每秒 626 条 |
| 依赖 | `torch>=2.14`，`transformers>=5.17`（含 `qwen3_5`） |
| 许可 | Apache-2.0 |

训练日志、完整报告和训练脚本在 [GitHub](https://github.com/leafiy/mia) 的 `logs/relabeled-mixed-neutral-qwen3.5-2b.log`、`reports/mia-qwen3.5-2b/` 和 `scripts/train-qwen-sentiment.py`。

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

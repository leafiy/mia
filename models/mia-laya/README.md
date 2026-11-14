---
license: apache-2.0
language:
- zh
pipeline_tag: text-classification
base_model: convaiinnovations/laya-multilingual
tags:
- sentiment-analysis
- sentiment-classification
- chinese-sentiment-analysis
- text-classification
- chinese
- lightweight
- 情感分析
- 中文情感分析
- 情感分类
- 文本分类
- 舆情分析
- mia
- laya
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

# Mia 中文情感分析轻量版 · mia-laya

判断一段中文是正面、负面还是中性，并给出三类概率。只有 3.2 亿参数，RTX 4090 上每秒近 2,000 条，一张 8G 显卡就能跑，适合大批量的评论和帖子。

Lightweight Chinese sentiment analysis model (positive / negative / neutral), 322M parameters, about 2,000 texts per second on an RTX 4090.

[GitHub](https://github.com/leafiy/mia) · [Hugging Face](https://huggingface.co/leafiy/mia-laya) · [ModelScope](https://modelscope.cn/models/leafiy2/mia-laya)

## 亮点

- **快。** RTX 4090 上每秒 1,977 条，是 Mia 三个情感版本里最快的。
- **准。** 30 行业 900 条上准确率 93.7%，比 2B 的两个版本低不到半个点；5,372 条留出测试上三分类 81.7%，同样能分中性的开源模型最高 69.3%。
- **省显存。** 一张 8G 显卡就能跑。

要最准、概率最可信，选 [mia-decider-2b](https://huggingface.co/leafiy/mia-decider-2b)。

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
pip install laya==0.3.20
hf download leafiy/mia-laya --local-dir mia-laya
# 或 modelscope download --model leafiy2/mia-laya --local_dir mia-laya
(cd mia-laya && sha256sum -c final.sha256)
```

```python
import json
from laya import load

agent = load('mia-laya/final', device='cuda')
questions = json.load(open('mia-laya/questions.json', encoding='utf-8'))
result = agent.predict_batch([{'text': '这家店的服务真的没话说'}], questions, batch_size=1)[0]
print(result['answers']['sentiment'])   # choice / probabilities / confidence
```

输入先做和训练时一样的归一化，用 [`normalize_text`](https://github.com/leafiy/mia/blob/main/scripts/prepare-laya-sentiment.py)。

## 用之前要知道

- 中性召回 0.770，低于 2B 的两个版本，错法多是把中性判成正面或负面。
- 训练时去掉了好评差评并存的文本，这类文本的输出没有定义。
- 只训练了三分类的 choice 头，返回里的 `act_probability` 没有训练过，别拿它做分流。
- 训练文本都在 100 字以内，长评论上准确率会下降。

## 模型信息

| | |
| --- | --- |
| 底座 | [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)（revision `e4e9ddf2`），编码器 jhu-clsp/mmBERT-base（22 层，hidden 768，256k 词表）+ Laya 两层判别头 |
| 参数量 | 322M |
| 输入 | Laya 的 `state`（`{"text": ...}`）加 `questions.json` 里的三选一问题 |
| 输出 | `负面 / 中性 / 正面`；温度 1.9488 已写进 `rl_agent_config.json`，`laya.load` 会自动应用 |
| 权重 | `final/model.safetensors`，fp32，约 1.3G；另有 `rl_agent_config.json`、`encoder/config.json`、`tokenizer/` |
| 训练数据 | 72,610 条中文短文本，好评差评并存的剔除 |
| 训练 | 3 轮，有效 batch 256，AdamW 编码器 2e-5 / 头 1e-4，类别权重 balanced，RTX 3080 约 25 分钟 |
| 留出测试 5,372 条 | 准确率 0.8172 · Macro-F1 0.8028 · 召回 负 0.794 / 中 0.770 / 正 0.894 |
| 30 行业 900 条 | 准确率 0.9367 · Macro-F1 0.9362 · 中性召回 0.833 |
| 速度 | RTX 4090，batch 64：每秒 1,977 条 |
| 依赖 | `laya==0.3.20`（当时环境 torch 2.14.0、transformers 5.17.0） |
| 许可 | Apache-2.0 |

训练日志、完整报告和训练脚本在 [GitHub](https://github.com/leafiy/mia) 的 `logs/relabeled-cw-balanced.log`、`reports/mia-laya/` 和 `scripts/train-laya-sentiment.py`。

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

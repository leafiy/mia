---
license: apache-2.0
language:
- zh
pipeline_tag: text-classification
base_model: Mapika/decider-2b
tags:
- sentiment-analysis
- sentiment-classification
- chinese-sentiment-analysis
- text-classification
- chinese
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

# Mia 中文情感分析 · mia-decider-2b

判断一段中文是正面、负面还是中性，并给出校准过的三类概率。从商品评论、微博帖子到问卷开放题和客服对话，都能直接用。这是 Mia 系列里最准的一个，推荐默认使用。

Chinese sentiment analysis model (positive / negative / neutral) with calibrated probabilities, for product reviews, social media posts, survey answers and customer-service chats.

[GitHub](https://github.com/leafiy/mia) · [Hugging Face](https://huggingface.co/leafiy/mia-decider-2b) · [ModelScope](https://modelscope.cn/models/leafiy2/mia-decider-2b)

## 亮点

- **分得出中性。** 5,372 条留出测试上三分类准确率 88.1%，同样能分中性的开源模型最高 69.3%。只要正负两类时准确率 97.8%。
- **换个行业照样准。** 覆盖 30 个行业、5 种文体的独立测试集（900 条）上 94.0%。
- **概率可信。** ECE 0.010，模型说有九成把握的判断，大约九成是对的。
- **8G 显卡就能跑。** 2B 参数，RTX 4090 上每秒 375 条。

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
pip install torch "transformers>=5" "decider-ai>=1.4"
hf download leafiy/mia-decider-2b --local-dir mia-decider-2b
# 或 modelscope download --model leafiy2/mia-decider-2b --local_dir mia-decider-2b
(cd mia-decider-2b && sha256sum -c final.sha256)
```

```python
import json
from decider.infer import Decider

d = Decider('mia-decider-2b/final')           # CUDA 上默认 bf16 + CUDA graphs
questions = json.load(open('mia-decider-2b/questions.json', encoding='utf-8'))
schema = d.schema(questions)                  # 题目只算一次，之后每条只算文本本身

texts = ['这家店的服务真的没话说', '本次记录涉及这家快递网点，相关信息来自页面与服务记录']
for t, r in zip(texts, schema.batch(texts)):
    print(t, r['answers']['sentiment']['choice'], r['answers']['sentiment']['probabilities'])
# 这家店的服务真的没话说 正面 {'负面': 0.0017, '中性': 0.004, '正面': 0.9943}
# 本次记录涉及这家快递网点，相关信息来自页面与服务记录 中性 {'负面': 0.0001, '中性': 0.9998, '正面': 0.0001}
```

- 输入先做和训练时一样的归一化，用 [`normalize_text`](https://github.com/leafiy/mia/blob/main/scripts/prepare-laya-sentiment.py)。
- 要起 HTTP 服务，decider 仓库的 `scripts/serve.sh mia-decider-2b/final 8000` 会提供 `POST /v1/systemone`。
- 它的底子是决策模型，同一份权重也能回答你自己写的选择题、是非题和打分题，这部分没有中文评测：

```python
d.system_one('快递三天没到，客服也不回', {
    'urgent': {'type': 'noul', 'instructions': '这条需要当天处理吗？'},
    'topic': {'type': 'choice', 'instructions': '主要在说什么？', 'criteria': {'物流': None, '质量': None, '服务': None}}})
```

## 用之前要知道

- 一条里好评差评都有时判为中性。只需要正负两类时，比较正面和负面的概率，别直接取三类里最大的那个。
- 训练文本都在 100 字以内，长评论上准确率会下降。
- 三类里负面召回最低（留出测试 0.851），漏掉的多是不带评价词的负面事实陈述。
- 只有 CUDA 上的题目缓存路径快；CPU 或 Apple 芯片上慢得多。

## 模型信息

| | |
| --- | --- |
| 底座 | [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b) v11（Hub revision `533964da`），Qwen3.5-2B-Base 经 decider 的监督训练、校准强化学习与 LoRA 阶段 |
| 参数量 | 1,882M |
| 输出 | `负面 / 中性 / 正面` 三类概率（`questions.json` 的 `criteria` 顺序）；温度 0.9989 已写进 `final/decider_config.json` |
| 权重 | `final/model.safetensors`，bf16，约 3.8G |
| 训练数据 | 76,112 条中文短文本，褒贬并存的归为中性；另混 15% decider 自带的通用决策题（13,432 条）防止遗忘 |
| 训练 | decider 的 `python -m decider.train`，1 轮 466 步，每步 32,768 token，lr 8e-6，RTX 4090 约 50 分钟 |
| 留出测试 5,372 条 | 准确率 0.8812 · Macro-F1 0.8705 · 召回 负 0.851 / 中 0.876 / 正 0.903 · ECE 0.0102 |
| 30 行业 900 条 | 准确率 0.9400 · Macro-F1 0.9397 |
| 通用判断 | decider 从没训过的 6 个英文领域：自定义题 0.932 → 0.935，路由题 0.895 → 0.927，没有遗忘 |
| 速度 | RTX 4090，bf16，batch 64：每秒 375 条 |
| 显存 | batch 16 约 6.6G，batch 64 约 19G |
| 依赖 | `torch`，`transformers>=5`（含 `qwen3_5`），`decider-ai>=1.4` 或 [decider](https://github.com/Mapika/decider) commit `a5120cc` |
| 许可 | Apache-2.0 |

训练日志、完整报告和评测脚本在 [GitHub](https://github.com/leafiy/mia) 的 `logs/mia-decider-2b.log`、`reports/mia-decider-2b/` 和 `scripts/`。

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

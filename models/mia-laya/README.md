# mia-laya

中文短文本情感三分类（负面 / 中性 / 正面）。在 Laya multilingual 判别模型上微调，走 `laya` SDK 加载，吞吐高、显存小。

| | |
| --- | --- |
| 底座 | [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)，revision `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`；编码器 jhu-clsp/mmBERT-base（22 层，hidden 768，256k 词表），Laya 两层判别头 |
| 参数量 | 322M |
| 权重 | 不在仓库里：从 [Hugging Face](https://huggingface.co/leafiy/mia-laya) 或 [ModelScope](https://modelscope.cn/models/leafiy2/mia-laya) 下载 `final/` 到本目录（`hf download leafiy/mia-laya --include "final/*" --local-dir models/mia-laya`），用 `final.sha256` 核对。含 fp32 `model.safetensors`（约 1.3G）、`rl_agent_config.json`、`encoder/config.json`、`tokenizer/` |
| 输入 | Laya 的 `state`（`{"text": ...}`）加 `questions.json` 里的三选一问题；答案取三个选项标记位的 logits |
| 标签顺序 | `负面, 中性, 正面` |
| 温度 | 1.9488，已写入 `rl_agent_config.json`，`laya.load` 会自动应用 |
| 训练数据 | 72,610 行中文短文本，大模型重标，褒贬混合剔除（未公开） |
| 训练 | 3 epochs，有效 batch 256，AdamW 编码器 2e-5 / 头 1e-4，6% warmup 余弦，类别权重 balanced（负 1.666 / 中 0.784 / 正 0.889），fp32 权重 + bf16 autocast，seed 42，RTX 3080 约 25 分钟 |
| 开发集 4,920 | Acc 0.8108 · Macro-F1 0.7974 |
| holdout 5,372 | Acc 0.8172 · Macro-F1 0.8028 · 召回 负 0.794 / 中 0.770 / 正 0.894 |
| 中文 900 条 | Acc 0.9367 · Macro-F1 0.9362 · 中性召回 0.833 |
| 公开数据集（二选一准确率） | ChnSentiCorp 0.8833 · online_shopping_10_cats 0.9186 · eprstmt 0.8820 · weibo_senti_100k 0.8030；DMSC 三分类 0.5422 |
| 对照：原版 Qwen3.5-2B 零样本 | holdout 0.6372 / Macro-F1 0.6283；中文 900 条 0.7944；ChnSentiCorp 0.8583 · online_shopping 0.8964 · eprstmt 0.8426 · weibo 0.8046（`reports/qwen3.5-2b-zero-shot/`） |
| 吞吐 | RTX 4090，batch 64，holdout 5,372 条端到端：约 1,977 条/秒（`reports/mia-laya/throughput.json`） |
| 许可 | 权重以 Apache-2.0 发布（与底座一致） |
| 依赖 | `laya==0.3.20`（当时环境 torch 2.14.0、transformers 5.17.0） |

```python
import json
from laya import load

agent = load('models/mia-laya/final', device='cuda')
questions = json.load(open('models/mia-laya/questions.json', encoding='utf-8'))
result = agent.predict_batch([{'text': '这家店的服务真的没话说'}], questions, batch_size=1)[0]
print(result['answers']['sentiment'])   # choice / probabilities / confidence
```

输入请先做与训练一致的归一化（`scripts/prepare-laya-sentiment.py` 的 `normalize_text`）。只监督了三分类的 choice 头，返回里的 `act_probability` 没有训练过，不要用它做分流。完整报告在 `reports/mia-laya/` 与 `reports/public/mia-laya.json`，训练日志 `logs/relabeled-cw-balanced.log`，训练脚本 `scripts/train-laya-sentiment.py`。

已知短板：中性召回 0.770 低于 `mia-qwen3.5-2b` 的 0.892，主要错法是把中性判成正面或负面；褒贬并存的文本没有定义好的输出（训练时被剔除）。评测标签来自大模型重标，不是人工金标。

## 和常见开源中文情感模型对比

| 模型 | 类别 | holdout 5,372 三分类 | 中文 900 条三分类 | holdout 正负二选一 | 900 条正负二选一 | ChnSentiCorp | 网购评论 | eprstmt | 微博 | DMSC Macro-F1 | 4090 每秒条数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [**mia-decider-2b**](https://huggingface.co/leafiy/mia-decider-2b) | 负 / 中 / 正 | **0.881** | 0.940 | **0.978** | **1.000** | 0.880 | 0.921 | 0.893 | **0.830** | **0.519** | 375 |
| [**mia-qwen3.5-2b**](https://huggingface.co/leafiy/mia-qwen3.5-2b) | 负 / 中 / 正 | 0.871 | **0.941** | 0.972 | **1.000** | 0.882 | 0.922 | 0.897 | 0.798 | 0.499 | 626 |
| [**mia-laya**](https://huggingface.co/leafiy/mia-laya) | 负 / 中 / 正 | 0.817 | 0.938 | 0.954 | 0.995 | 0.883 | 0.919 | 0.882 | 0.803 | 0.467 | 1,977 |
| [Erlangshen-Roberta-330M-Sentiment](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment) | 负 / 正 | 0.452 | 0.656 | 0.880 | 0.983 | **0.968** | **0.989** | **0.997** | 0.729 | 0.415 | 1,378 |
| [Erlangshen-Roberta-110M-Sentiment](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment) | 负 / 正 | 0.448 | 0.646 | 0.871 | 0.968 | 0.962 | 0.979 | 0.982 | 0.721 | 0.409 | 3,809 |
| [uer/roberta-base-finetuned-dianping-chinese](https://huggingface.co/uer/roberta-base-finetuned-dianping-chinese) | 负 / 正 | 0.455 | 0.657 | 0.885 | 0.985 | 0.877 | 0.894 | 0.795 | 0.726 | 0.414 | 3,812 |
| [uer/roberta-base-finetuned-jd-binary-chinese](https://huggingface.co/uer/roberta-base-finetuned-jd-binary-chinese) | 负 / 正 | 0.431 | 0.654 | 0.839 | 0.982 | 0.834 | 0.928 | 0.902 | 0.704 | 0.455 | 3,795 |
| [cardiffnlp/twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) | 负 / 中 / 正 | 0.693 | 0.664 | 0.921 | 0.750 | 0.809 | 0.894 | 0.848 | 0.813 | 0.397 | 3,650 |
| [lxyuan/distilbert-base-multilingual-cased-sentiments-student](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student) | 负 / 中 / 正 | 0.486 | 0.560 | 0.885 | 0.817 | 0.852 | 0.903 | 0.864 | 0.797 | 0.447 | 6,699 |
| [tabularisai/multilingual-sentiment-analysis](https://huggingface.co/tabularisai/multilingual-sentiment-analysis) | 五档合为三类 | 0.622 | 0.777 | 0.837 | 0.785 | 0.822 | 0.868 | 0.841 | 0.668 | 0.374 | 6,556 |

同一套评测、同一台 RTX 4090（batch 64），脚本 `scripts/evaluate-public-benchmarks.py`（开源模型用 `--model-type hf`）与 `scripts/benchmark-throughput.py`，逐模型报告在 GitHub 仓库的 `reports/open-models/`。

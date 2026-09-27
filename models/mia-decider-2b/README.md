# mia-decider-2b

中文短文本情感三分类（负面 / 中性 / 正面）。在开源决策模型 decider-2b 上用 Mia 的数据接着全量微调一轮，褒贬并存的文本按中性处理。它仍是一个 Jev 式（System One）决策模型：情感是其中一道选择题，同一份权重也能回答别的选择题、是非题和打分题。

| | |
| --- | --- |
| 底座 | [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b) v11（Hub revision `533964da`）：Qwen/Qwen3.5-2B-Base 经 decider 的监督训练、校准强化学习与 LoRA 阶段；Apache-2.0，独立项目 |
| 参数量 | 1,882M（`Qwen3_5ForCausalLM`，答案读选项字母在 `lm_head` 上的 logits） |
| 权重 | 不在仓库里：从 [Hugging Face](https://huggingface.co/leafiy/mia-decider-2b) 或 [ModelScope](https://modelscope.cn/models/leafiy2/mia-decider-2b) 下载 `final/` 到本目录（`hf download leafiy/mia-decider-2b --include "final/*" --local-dir models/mia-decider-2b`），用 `final.sha256` 核对。`final/model.safetensors` 为 bf16，约 3.8G；`final/decider_config.json` 存温度、题面缓存开关和训练记录 |
| 题目 | `questions.json`：一道 Choice，「判断文本作者表达的整体情感倾向。」，三个选项带定义；训练和推理用的是同一道题 |
| 标签顺序 | `负面, 中性, 正面`（`questions.json` 的 `criteria` 顺序） |
| 温度 | 0.9989（在 calibration 切分上拟合，`decider_config.json` 的 `temperature` 与 `temperature_by_type.choice`） |
| 打分路径 | 默认题面在前、只算一次并缓存（`schema_first: true`，`Decider.schema(questions).batch(texts)`）；也可以走题面在后、每条重算的路径，结果见下 |
| 训练数据 | Mia 的 76,112 行中文短文本（大模型重标，褒贬混合并入中性，未公开）+ 13,432 行 decider 自带 `teacher_data/` 的通用决策题（15%） |
| 训练 | decider 自己的 `python -m decider.train`，`delta` 配方改两处（warmup 50、`none_prob 0`）；1 epoch，466 步，每步 32,768 token，lr 8e-6，一半样本题面在前，bf16，seed 0，RTX 4090 约 50 分钟 |
| 开发集 4,920 | Acc 0.8866 · Macro-F1 0.8763 · 中性召回 0.888 |
| holdout 5,372 | Acc 0.8812 · Macro-F1 0.8705 · 召回 负 0.851 / 中 0.876 / 正 0.903 · ECE 0.0102 |
| 中文 900 条 | Acc 0.9400 · Macro-F1 0.9397 · 中性召回 0.840 |
| 公开数据集（二选一准确率） | ChnSentiCorp 0.8800 · online_shopping_10_cats 0.9208 · eprstmt 0.8934 · weibo_senti_100k 0.8302；DMSC 三分类 0.5352 |
| 题面在后的路径 | 开发集 0.8831 · holdout 0.8762 · 中文 900 条 0.9333 · 公开 0.8867 / 0.9286 / 0.9049 / 0.8250 · DMSC 0.5330（`reports/mia-decider-2b/state-first-*`） |
| 对照：原版 decider-2b 零样本 | 同一道题：holdout 0.8438 / Macro-F1 0.8320；中文 900 条 0.9078，中性召回 0.730（`reports/decider-2b-zero-shot/`） |
| 通用判断有没有丢 | decider 从没训过的 6 个领域（英文）：自定义题 0.932 → 0.935，路由题 0.895 → 0.927（`reports/mia-decider-2b/history.json`） |
| 吞吐 | RTX 4090，bf16，batch 64，holdout 端到端：题面缓存 375 条/秒，题面在后 129 条/秒 |
| 显存 | 题面缓存路径整卡占用：batch 16 约 6.6G，batch 64 约 19G（`reports/mia-decider-2b/gpu-memory.json`）；8G 卡请把 batch 调到 16 |
| 许可 | 权重以 Apache-2.0 发布（与底座一致） |
| 依赖 | `torch`，`transformers>=5`（含 `qwen3_5`），[decider](https://github.com/Mapika/decider)（commit `a5120cc`，或 `decider-ai>=1.4`） |

```python
import json
from decider.infer import Decider

d = Decider('models/mia-decider-2b/final')
questions = json.load(open('models/mia-decider-2b/questions.json', encoding='utf-8'))
schema = d.schema(questions)
for r in schema.batch(['这家店的服务真的没话说']):
    print(r['answers']['sentiment'])   # choice 正面，probabilities {负面: 0.0017, 中性: 0.004, 正面: 0.9943}
```

输入请先做与训练一致的归一化（`scripts/prepare-laya-sentiment.py` 的 `normalize_text`）。在只有正负两类的场景里，请比较正、负两类的概率，不要用三类 argmax：它会把两成到三成的褒贬并存或纯陈述文本判成中性。完整报告在 `reports/mia-decider-2b/` 与 `reports/public/mia-decider-2b.json`，训练日志 `logs/mia-decider-2b.log`，数据构建脚本 `scripts/prepare-decider-sentiment.py`，评测脚本 `scripts/evaluate-systemone-sentiment.py`。

已知短板：中文 900 条上 300 条中性有 48 条判成负面，比 `mia-qwen3.5-2b` 的 29 条多；褒贬并存的文本会被判为中性，这是训练口径而不是缺陷；训练文本不超过 100 字，长评论上表现会下降；中文自定义题目能答，但没有评测集；只有 CUDA 上的题面缓存路径快。评测标签来自大模型重标，不是人工金标。

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

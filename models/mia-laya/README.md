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

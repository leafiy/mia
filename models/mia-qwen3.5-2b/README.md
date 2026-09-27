# mia-qwen3.5-2b

中文短文本情感三分类（负面 / 中性 / 正面）。Qwen3.5-2B 的文本主干加一个线性分类头，全量微调，褒贬并存的文本按中性处理。

| | |
| --- | --- |
| 底座 | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)，只保留 `model.language_model.*`（24 层，hidden 2048，线性注意力 + 每 4 层一层全注意力），丢掉视觉塔、`lm_head` 与 MTP 头 |
| 参数量 | 1,882M（含 248,320 词表的嵌入） |
| 权重 | 不在仓库里：从 [Hugging Face](https://huggingface.co/leafiy/mia-qwen3.5-2b) 或 [ModelScope](https://modelscope.cn/models/leafiy2/mia-qwen3.5-2b) 下载 `final/` 到本目录（`hf download leafiy/mia-qwen3.5-2b --include "final/*" --local-dir models/mia-qwen3.5-2b`），用 `final.sha256` 核对。`final/model.safetensors` 为 bf16，约 3.6G；`final/config.json` 是 `Qwen3_5TextConfig`，`num_labels=3` |
| 输入模板 | `文本：{text}\n情感倾向：`，右侧 padding，不加特殊 token；取最后一个非 pad token 的隐状态 |
| 标签顺序 | `负面, 中性, 正面`（`sentiment-config.json` 的 `labels`） |
| 温度 | 1.7063，推理时 logits 先除以它再 softmax |
| 训练数据 | 76,112 行中文短文本，大模型重标，褒贬混合并入中性（未公开） |
| 训练 | 2 epochs，有效 batch 256，AdamW 主干 2e-5 / 头 1e-4，6% warmup 余弦，fp32 权重 + bf16 autocast，seed 42，RTX 4090 约 40 分钟 |
| 开发集 4,920 | Acc 0.8689 · Macro-F1 0.8543 |
| holdout 5,372 | Acc 0.8710 · Macro-F1 0.8564 · 召回 负 0.774 / 中 0.892 / 正 0.889 |
| 中文 900 条 | Acc 0.9411 · Macro-F1 0.9411 · 中性召回 0.903 |
| 公开数据集（二选一准确率） | ChnSentiCorp 0.8817 · online_shopping_10_cats 0.9224 · eprstmt 0.8967 · weibo_senti_100k 0.7976；DMSC 三分类 0.5120 |
| 对照：原版 Qwen3.5-2B 零样本 | 同一口径提示词、受限选择：holdout 0.6372 / Macro-F1 0.6283 / 中性召回 0.351；中文 900 条 0.7944；ChnSentiCorp 0.8583 · online_shopping 0.8964 · eprstmt 0.8426 · weibo 0.8046（`reports/qwen3.5-2b-zero-shot/`、`reports/public/qwen3.5-2b-zero-shot.json`） |
| 吞吐 | RTX 4090，bf16，batch 64，PyTorch 参考实现的线性注意力，holdout 5,372 条端到端：约 626 条/秒（`reports/mia-qwen3.5-2b/throughput.json`） |
| 许可 | 权重以 Apache-2.0 发布（与底座一致） |
| 依赖 | `torch>=2.14`，`transformers>=5.17`（含 `qwen3_5`） |

```python
import json, torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import Qwen3_5TextForSequenceClassification

final = 'models/mia-qwen3.5-2b/final'
cfg = json.load(open(f'{final}/sentiment-config.json', encoding='utf-8'))
tok = AutoTokenizer.from_pretrained(final)
model = Qwen3_5TextForSequenceClassification.from_pretrained(final, dtype=torch.bfloat16).cuda().eval()
enc = tok([cfg['template'].format(text='这家店的服务真的没话说')], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
with torch.inference_mode():
    probs = (model(**enc).logits.float() / cfg['temperature']).softmax(-1)[0]
print(dict(zip(cfg['labels'], probs.tolist())))
```

输入请先做与训练一致的归一化（`scripts/prepare-laya-sentiment.py` 的 `normalize_text`）。在只有正负两类的场景里，请比较正、负两类的概率，不要用三类 argmax：它会把两成到三成的褒贬并存或纯陈述文本判成中性。完整报告在 `reports/mia-qwen3.5-2b/` 与 `reports/public/mia-qwen3.5-2b.json`，训练日志 `logs/relabeled-mixed-neutral-qwen3.5-2b.log`，训练脚本 `scripts/train-qwen-sentiment.py`。

已知短板：负面召回 0.774 是三类里最低的，错的多是不带评价词的负面事实陈述；褒贬并存的文本会被判为中性，这是训练口径而不是缺陷；训练文本不超过 100 字，长评论上表现会下降。评测标签来自大模型重标，不是人工金标。

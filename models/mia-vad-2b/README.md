---
license: apache-2.0
language:
- zh
pipeline_tag: text-classification
base_model: Mapika/decider-2b
tags:
- emotion-recognition
- emotion-classification
- emotion-analysis
- sentiment-analysis
- valence-arousal-dominance
- vad
- chinese
- 情绪识别
- 情绪分析
- 情感分析
- 多维情感分析
- 中文情感分析
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
- pearsonr
---

# Mia 中文情绪识别 · mia-vad-2b（效价、唤醒度、七类情绪）

一条中文一次给出六项情绪判断，每项都是校准过的概率：

- 整体态度：负面 / 中性 / 正面，和 [mia-decider-2b](https://modelscope.cn/models/leafiy2/mia-decider-2b) 是同一道题
- 效价 0～6 分（3 是没有倾向，减 3 得 −3～+3）
- 唤醒度 0～4 分
- 支配度 0～4 分
- 主导情绪：喜悦、愤怒、悲伤、恐惧、惊讶、厌恶、无明显情绪
- 情绪强度 0～3 分

Chinese emotion recognition model: overall sentiment, valence, arousal, dominance (VAD), seven emotion categories and intensity in one pass.

[GitHub](https://github.com/leafiy/mia) · [ModelScope](https://modelscope.cn/models/leafiy2/mia-vad-2b)

## 亮点

- **公开评测上超过 27B 大模型。** 人工打分的中文情绪语料 CVAT 上效价相关 0.843（Qwen3.8-27B 0.740），SMP2020-EWECT 微博情绪分类准确率 0.711（27B 0.689），参数只有它的十几分之一。
- **三分类不打折。** 留出测试 5,372 条上准确率 0.8816，和只做三分类的 mia-decider-2b（0.8812）持平。
- **一次前向，六项全出。** 不用为每一项单独跑一个模型。

## 效果

金标 2,000 条由第三个大模型按同一份书面口径（[rubric.md](https://github.com/leafiy/mia/blob/main/models/mia-vad-2b/rubric.md)）标注：1,454 条随机抽取（前五行用其中可用的 1,439 条），546 条按稀有情绪补抽（Macro-F1 用全部 1,981 条可用的）。27B 是训练标签的标注者之一，稀有部分也按它的判断挑，它在金标上的一致度偏高，只作参照。

| | `mia-vad-2b` | `mia-decider-2b` 零样本 | Qwen3.8-27B |
| --- | ---: | ---: | ---: |
| 效价 r | 0.881 | 0.845 | **0.908** |
| 唤醒度 r | **0.845** | 0.791 | 0.813 |
| 支配度 r | **0.600** | 0.459 | 0.574 |
| 强度 r | **0.817** | 0.735 | 0.795 |
| 主导情绪准确率 | 0.792 | 0.737 | **0.810** |
| 主导情绪 Macro-F1 | 0.671 | 0.500 | **0.707** |

两份训练和挑选时都没碰过的公开评测：

| | `mia-vad-2b` | `mia-decider-2b` 零样本 | Qwen3.8-27B |
| --- | ---: | ---: | ---: |
| CVAT 2,931 句，效价 r | 0.843 | **0.857** | 0.740 |
| CVAT，唤醒度 r | **0.549** | 0.522 | 0.486 |
| CVAT 新闻 1,471 句，效价 r | 0.791 | **0.826** | 0.635 |
| SMP2020-EWECT 微博 4,998 条，准确率 | **0.711** | 0.696 | 0.689 |
| SMP2020-EWECT，Macro-F1 | **0.698** | 0.628 | 0.678 |
| SMP2020-EWECT，惊讶 F1 | 0.561 | 0.254 | **0.568** |

CVAT 是人工 1～9 分的效价和唤醒度评分，相关系数用模型各档概率的期望（T=1）算。EWECT 没有厌恶类，判成厌恶的算错，27B 只算它判为可用的 4,920 条。

| 三分类 | |
| --- | --- |
| 留出测试 5,372 条 | 准确率 0.8816 · Macro-F1 0.8702 · ECE 0.0106（mia-decider-2b：0.8812 / 0.8705 / 0.0102） |
| 开发集 4,920 条 | 准确率 0.8890 · ECE 0.0086 |
| 30 行业 900 条 | 准确率 0.9367（mia-decider-2b：0.9400） |

主导情绪各类 F1：喜悦 0.85、无明显情绪 0.77、悲伤 0.73、恐惧 0.65、厌恶 0.57、愤怒 0.57、惊讶 0.56。

## 快速开始

```bash
pip install torch "transformers>=5" "decider-ai>=1.4" modelscope
modelscope download --model leafiy2/mia-vad-2b --local_dir mia-vad-2b
(cd mia-vad-2b && sha256sum -c final.sha256)
```

```python
import json
from decider.infer import Decider
from decider.schema_engine import SchemaEngine

d = Decider('mia-vad-2b/final')
d._se = SchemaEngine(d.eng, use_graphs=False, chat=d.chat)   # 六道题各带一份题面缓存，按（批大小, 长度）抓 CUDA 图会占满显存
questions = json.load(open('mia-vad-2b/questions.json', encoding='utf-8'))
answers = d.schema(questions).batch(['这家店的服务真的没话说'])[0]['answers']

w = {k: p ** (1.04 / 1.2) for k, p in answers['emotion']['probabilities'].items()}   # 主导情绪换到它自己的温度 1.2
emotion = {k: v / sum(w.values()) for k, v in w.items()}
valence = sum(int(k) * p for k, p in answers['valence']['probabilities'].items()) - 3   # 各档概率的期望，−3～+3
print(answers['sentiment']['probabilities'], max(emotion, key=emotion.get), round(valence, 2))
```

- **打分题取期望。** 用各档概率的期望当分数，上面的相关系数都这样算。唤醒度、支配度、强度的取法和效价一样，只是不减 3。
- **主导情绪要再换一次温度。** decider 的温度只按题型分，三分类和主导情绪都是选择题，配置里的 choice 温度是三分类的 1.04。主导情绪的概率请按 p^(1.04/1.2) 换算再归一，`decider_config.json` 的 `emotion_temperature` 记着 1.2。
- **先归一化。** 输入先做和训练时一样的归一化（[`normalize_text`](https://github.com/leafiy/mia/blob/main/scripts/prepare-laya-sentiment.py)），再截到 512 字。
- **CUDA 图要看显存。** 去掉 `d._se = ...` 那一行就走默认的 CUDA 图，结果不变。它按（批大小，长度档）各抓一张图，每张都带一份六道题的前缀缓存，长短不一的文本一多，48G 显卡也会占满。

## 用之前要知道

- **惊讶找得最不全。** 金标上查全只有 0.44，多被判成无明显情绪或喜悦。恐惧的查全 0.56，厌恶 0.54。
- **支配度最弱。** 和金标的相关只有 0.60，几个大模型标注者之间也只有 0.54～0.58，这一项本身就难判。
- **客观报道的负面新闻，效价判中性。** CVAT 的评分人按事件本身打分，所以新闻句上效价相关是 0.79，评论类文本是 0.78～0.94。
- **好评差评并存时，三分类判中性。** 这和 mia-decider-2b 一致；效价和主导情绪按占上风的一边给。

## 模型信息

| | |
| --- | --- |
| 底座 | [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b) v11（Hub revision `533964da`），Qwen3.5-2B-Base 经 decider 的监督训练、校准强化学习与 LoRA 阶段 |
| 参数量 | 1,882M |
| 题目 | `questions.json`：六道题，`sentiment`、`emotion` 是选择题，`valence`、`arousal`、`dominance`、`intensity` 是打分题；训练和推理用的是同一份 |
| 温度 | 三分类 1.04，打分题 0.88，主导情绪 1.2 |
| 权重 | `final/model.safetensors`，bf16，约 3.8G；`final/decider_config.json` 存温度和训练记录 |
| 训练数据 | Mia 的 76,112 条三分类数据；42,000 条带六项标签的新中文短文本（社媒、电商、文学、医疗等），稀有情绪重复后 48,578 条；decider 自带的通用决策题 22,004 条（15%） |
| 新文本的标签 | 三个大模型按同一份口径标注、复核和定向纠正；主导情绪两家一致才学，三分类只学与 mia-decider-2b 一致的 |
| 训练 | decider 的 `python -m decider.train`，1 轮 1,569 步，每步 32,768 token，lr 8e-6，warmup 180，RTX 4090 约 2 小时 |
| 通用判断 | decider 的两组英文留出题：自定义题 0.932、路由题 0.932（mia-decider-2b 0.935 / 0.927） |
| 速度 | RTX 4090，bf16，batch 32，CUDA 图：67 条/秒（每道题一行，六道题算六行） |
| 依赖 | `torch`，`transformers>=5`（含 `qwen3_5`），`decider-ai>=1.4`（训练与评测用 1.5.0） |
| 许可 | Apache-2.0 |

完整报告在 [GitHub](https://github.com/leafiy/mia) 的 `reports/mia-vad-2b/`（`eval.json` 验收评测，`analysis.json` 校准、分类别与外部评测，`learning-curve.json` 各次训练对比，`training-config.json` 训练参数与数据构成），训练日志 `logs/mia-vad-2b.log`。两份外部评测可以用 `scripts/evaluate-mia-vad-external.py` 重算，数据需自行从原发布处下载。

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

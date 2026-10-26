# mia-vad-2b

中文短文本的多维情绪判断。一条文本一次给出六项，每项都是校准过的概率：

- 整体态度：负面 / 中性 / 正面，题目和 [mia-decider-2b](https://modelscope.cn/models/leafiy2/mia-decider-2b) 是同一道
- 效价 0～6 分（3 是没有倾向）
- 唤醒度 0～4 分
- 支配度 0～4 分
- 主导情绪：喜悦、愤怒、悲伤、恐惧、惊讶、厌恶、无明显情绪
- 情绪强度 0～3 分

它是 Mia 的多维版。底座同样是开源决策模型 decider-2b v11，在 Mia 的三分类数据之外加了 42,000 条带六项标签的新文本，从底座重新训练一轮。三分类的成绩和 mia-decider-2b 持平。

| | |
| --- | --- |
| 底座 | [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b) v11（Hub revision `533964da`）：Qwen/Qwen3.5-2B-Base 经 decider 的监督训练、校准强化学习与 LoRA 阶段；Apache-2.0，独立项目 |
| 参数量 | 1,882M（`Qwen3_5ForCausalLM`，答案读选项字母在 `lm_head` 上的 logits） |
| 权重 | 不在仓库里：从 [ModelScope](https://modelscope.cn/models/leafiy2/mia-vad-2b) 下载 `final/` 到本目录（`modelscope download --model leafiy2/mia-vad-2b --include "final/*" --local_dir models/mia-vad-2b`），用 `final.sha256` 核对。`final/model.safetensors` 为 bf16，约 3.8G；`final/decider_config.json` 存温度和训练记录 |
| 标注口径 | `rubric.md`：八个字段怎么判、容易分歧的情况、字段之间的自洽规则，三个大模型标注者用的是同一份 |
| 题目 | `questions.json`：六道题，`sentiment`、`emotion` 是选择题，`valence`、`arousal`、`dominance`、`intensity` 是打分题；训练和推理用的是同一份 |
| 温度 | 三分类 1.04（在 Mia 的 calibration 4,926 条上拟合），打分题 0.88，主导情绪 1.2（后两项在新文本 calibration 4,000 条上拟合） |
| 训练数据 | Mia 的 76,112 条三分类；新文本 42,000 条（社媒、电商、文学、医疗等中文短文本，未公开），稀有情绪重复后 48,578 条；decider 自带 `teacher_data/` 的通用决策题 22,004 条（15%） |
| 新文本的标签 | Qwen3.8-27B 按书面口径标全部候选；第二个大模型重标随机 20% 和 Qwen 判出稀有情绪的行；第三个大模型标 2,000 条金标，再定向纠正 3,000 条（27B 判出稀有情绪的、两家分歧最大的、和 mia-decider-2b 三分类相反的各 1,000 条）。主导情绪两家一致才学，三分类只学与 mia-decider-2b 判断一致的 |
| 训练 | decider 自己的 `python -m decider.train`，1 epoch，1,569 步，每步 32,768 token，lr 8e-6，warmup 180，一半样本题面在前，`none_prob 0`，bf16，seed 0，RTX 4090 约 2 小时 |
| 依赖 | `torch`，`transformers>=5`（含 `qwen3_5`），`decider-ai>=1.4`（训练与评测用 1.5.0） |

## 评测

金标 2,000 条由第三个大模型按同一份书面口径标注：1,454 条随机抽取，另 546 条按稀有情绪补抽。

| | |
| --- | --- |
| 三分类 holdout 5,372 | Acc 0.8816 · Macro-F1 0.8702 · ECE 0.0106（mia-decider-2b：0.8812 / 0.8705 / 0.0102） |
| 三分类开发集 4,920 | Acc 0.8890 · ECE 0.0086 |
| 三分类中文 900 条 | Acc 0.9367（mia-decider-2b：0.9400） |
| 效价 / 唤醒度 / 支配度 / 强度 | 和金标随机部分 1,439 条的 Pearson r：0.881 / 0.845 / 0.600 / 0.817 |
| 主导情绪 | 金标随机部分准确率 0.792；全部 1,981 条 Macro-F1 0.671（喜悦 0.85、无明显情绪 0.77、悲伤 0.73、恐惧 0.65、厌恶 0.57、愤怒 0.57、惊讶 0.56） |
| CVAT 2,931 句（人工 1～9 分） | 效价 r 0.843，唤醒度 r 0.549 |
| SMP2020-EWECT 微博 usual 测试集 4,998 条 | 准确率 0.711，Macro-F1 0.698（这份数据没有厌恶类，判成厌恶的算错） |
| 通用判断有没有丢 | decider 的两组英文留出题：自定义题 0.932、路由题 0.932（mia-decider-2b 0.935 / 0.927） |
| 吞吐 | RTX 4090，bf16，batch 32，CUDA graph：67 条/秒（每道题一行，六道题算六行） |

## 用法

```python
import json
from decider.infer import Decider

d = Decider('models/mia-vad-2b/final')
questions = json.load(open('models/mia-vad-2b/questions.json', encoding='utf-8'))
schema = d.schema(questions)
answers = schema.batch(['这家店的服务真的没话说'])[0]['answers']
answers['emotion']['probabilities']   # {'喜悦': ..., '愤怒': ..., ...}
answers['valence']['probabilities']   # {'0': ..., ..., '6': ...}
```

- **打分题取期望。** 用各档概率的期望当分数，上面的相关系数都这样算；效价减 3 得 −3～+3。
- **主导情绪要再换一次温度。** decider 的温度只按题型分，三分类和主导情绪都是选择题，配置里的 choice 温度是三分类的 1.04。主导情绪的概率请按 p^(1.04/1.2) 换算再归一，`decider_config.json` 的 `emotion_temperature` 记着 1.2。
- **先归一化。** 输入请先做和 Mia 训练一致的归一化（`scripts/prepare-laya-sentiment.py` 的 `normalize_text`），再截到 512 字。
- **长短不一的大批量用 eager。** 默认走 CUDA 图，每个（批大小，长度档）抓一张图，每张都带一份六道题的前缀缓存。长短不一的文本一多，48G 显卡也会占满。这时在 `d.schema(...)` 之前设 `from decider.schema_engine import SchemaEngine; d._se = SchemaEngine(d.eng, use_graphs=False, chat=d.chat)`，结果不变。

## 已知短板

- **惊讶找得最不全。** 金标上查全只有 0.44，多被判成无明显情绪或喜悦。恐惧的查全 0.56，厌恶 0.54。
- **支配度最弱。** 和金标的相关只有 0.60，几个大模型标注者之间也只有 0.54～0.58，这一项本身就难判。
- **客观报道的负面新闻判中性效价。** 这是口径如此。CVAT 的评分人按事件本身打分，所以新闻句上效价相关是 0.79，评论类文本是 0.78～0.94。
- **褒贬并存的三分类判中性。** 这和 mia-decider-2b 的口径一致；效价和主导情绪则按占上风的一边给。

完整报告在 `reports/mia-vad-2b/`（`eval.json` 验收评测，`analysis.json` 校准、分类别与外部考卷，`learning-curve.json` 各次训练对比，`training-config.json` 训练参数与数据构成），训练日志 `logs/mia-vad-2b.log`。外部考卷可以用 `scripts/evaluate-mia-vad-external.py` 重算，两份数据需自行从原发布处下载。

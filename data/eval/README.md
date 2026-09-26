# 评测数据

这里是 Mia 两个版本和全部对照实验共用的评测集。训练集不在仓库里。

| 文件 | 行数 | 标签 负 / 中 / 正 | 用途 |
| --- | ---: | --- | --- |
| `test.jsonl` | 4,920 | 838 / 2,391 / 1,691 | 开发集。训练时每个 epoch 评一次、选模型 |
| `unseen-test.jsonl` | 5,372 | 893 / 2,612 / 1,867 | holdout。只在最后评一次，主 README 的排名以它为准 |
| `calibration.jsonl` | 4,926 | 808 / 2,148 / 1,970 | 拟合温度，不参与训练与选模型 |
| `中文情感测试集_01.csv` | 300 | 100 / 100 / 100 | 独立标注的模板化基准，30 个行业 × 5 种文体 |
| `中文情感测试集_02.csv` | 300 | 100 / 100 / 100 | 同上 |
| `中文情感测试集_03.csv` | 300 | 100 / 100 / 100 | 同上 |
| `relabel-raw.jsonl` | 3,091 批 | 五选一原始标签 | 重标的原始回答：每批 30 个行 id 及大模型给出的标签，不含文本 |
| `manifest.json` | | | 各切分的行数、SHA-256、标签分布；重标的提示词哈希、token 用量、早期标签→新标签转移矩阵 |

公开数据集（ChnSentiCorp、online_shopping_10_cats、eprstmt、weibo_senti_100k、DMSC）不放在仓库里，`scripts/evaluate-public-benchmarks.py` 运行时从 Hugging Face Hub 下载到 `data/public-cache/`。

## 三个 JSONL 的格式

Laya notebook 兼容格式，`state`、`questions`、`gold` 三个字段都是 JSON 字符串：

```json
{"id": "sent-341086",
 "state": "{\"text\": \"归一化后的文本\"}",
 "questions": "{\"sentiment\": {\"type\": \"choice\", \"instructions\": \"判断文本表达的整体情感倾向。\", \"criteria\": {\"负面\": \"表达不满、失望、批评等负面态度\", \"中性\": \"态度客观或情绪不明显\", \"正面\": \"表达满意、赞赏、愉快等正面态度\"}}}",
 "gold": "{\"sentiment\": {\"type\": \"choice\", \"label\": \"中性\", \"probabilities\": {\"负面\": 0.0, \"中性\": 1.0, \"正面\": 0.0}}}"}
```

文本已经过与训练相同的归一化（NFKC、去 HTML / URL / @提及 / 表情与装饰符号、合并重复标点，5 到 100 个 Unicode 字符）。`id` 是源语料的行号，用它可以在 `relabel-raw.jsonl` 里找到该行的重标结果。

## 标签是怎么来的

- 三个 JSONL 的 `gold` 是大模型重标的结果（五选一：正面 / 负面 / 中性 / 褒贬混合 / 无法判断，temperature 0，关闭思考，每批 30 条随机混排），**褒贬混合写成中性，无法判断的行剔除**。`manifest.json` 的 `relabel` 节里有提示词哈希与每个切分的处理计数。
- 三份 CSV 的标签由构造者给出，与重标无关；它把褒贬并存的过程记录标为中性，这也是我们最终把褒贬混合并入中性的依据。
- 三个 JSONL 与训练集来自不同的来源批次（按批次哈希切分，同一批次不跨切分）；三份 CSV 与训练文本的精确重叠为 0。
- 三个 JSONL 做过个人信息扫描（手机号、证件号、邮箱、URL、即时通讯号）并剔除了命中的行，以及少量点名具体产品或平台的行，共 8 行；`manifest.json` 的 `public_eval_note` 记录了各切分剔除的数量。

## 与训练集的关系

`manifest.json` 里列出的 `train.jsonl`（76,112 行）没有随仓库发布，评测 JSONL 的行 id 与它不重叠。

#!/usr/bin/env python3
"""Generate docs/charts/*.svg and docs/index.html for the Mia release from reports/summary.json (stdlib only)."""
import html
import re
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '/root/1/mia')
SUMMARY = json.loads((ROOT / 'reports' / 'summary.json').read_text(encoding='utf-8'))
RUNS = SUMMARY['runs']
ZS = SUMMARY.get('zero_shot')            # the untouched Qwen3.5-2B scored zero-shot; no training-side fields
ALL = ([ZS] if ZS else []) + RUNS
BY = {r['alias']: r for r in ALL}
LABELS = SUMMARY['label_order']
CHARTS = ROOT / 'docs' / 'charts'
CHARTS.mkdir(parents=True, exist_ok=True)

FONT = "font-family='-apple-system, BlinkMacSystemFont, \"Segoe UI\", \"PingFang SC\", \"Microsoft YaHei\", \"Noto Sans CJK SC\", sans-serif'"
INK, MUTED, GRID = '#1f2328', '#57606a', '#d0d7de'
def _rows(key):
    m = re.search(r'\(([\d,]+) rows\)', SUMMARY['unified_eval'][key])
    return m.group(1) if m else ''

SETS = [('unified_dev', f'开发集 {_rows("dev")}', '#0969da'), ('unified_holdout', f'holdout {_rows("holdout")}', '#8250df'), ('csv', '中文 CSV 900', '#bf8700')]
CLASS_COLORS = {'负面': '#cf222e', '中性': '#6e7781', '正面': '#1a7f37'}
MIA = {'mia-laya', 'mia-qwen3.5-2b'}


def metric(run, key, field):
    m = run['chinese_csv']['pooled'] if key == 'csv' else run[key]
    return m['per_class']['中性']['recall'] if field == 'neutral_recall' else m[field]


def esc(s):
    return html.escape(str(s), quote=True)


def svg_open(w, h, title):
    return [f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' role='img' aria-label='{esc(title)}' {FONT}>",
            f"<rect width='{w}' height='{h}' fill='#ffffff'/>",
            f"<text x='24' y='30' font-size='18' font-weight='600' fill='{INK}'>{esc(title)}</text>"]


def wrap_label(text, width=16):
    """Greedy wrap for mixed CJK/Latin labels: break after a space or closing punctuation, or before an opening bracket."""
    after, before = ' ，、）)/-', '（('
    lines, current = [], ''
    for ch in text:
        current += ch
        if len(current) >= width:
            candidates = [i + 1 for i, c in enumerate(current) if c in after] + [i for i, c in enumerate(current) if c in before and i > 0]
            cut = max((c for c in candidates if c >= width // 2), default=len(current))
            lines.append(current[:cut].rstrip())
            current = current[cut:].lstrip()
    if current:
        lines.append(current)
    return [l for l in lines if l] or ['']


def hbar_chart(path, title, runs, value_fn, series, xmin, xmax, note=None, fmt='{:.3f}'):
    """Grouped horizontal bars: one row per run, one bar per series. Long row labels wrap; the legend wraps; the note sits below it."""
    bar_h, gap, line_h = 18, 4, 16
    labels = [wrap_label(run['alias']) for run in runs]
    row_hs = [max(22 * len(series) + 14, line_h * len(ls) + 14) for ls in labels]
    longest = max(len(l) for ls in labels for l in ls)
    left = min(340, 40 + int(longest * 13.5))
    right, top = 90, 56
    w = 1080
    plot_w = w - left - right
    def x(v):
        return left + (v - xmin) / (xmax - xmin) * plot_w
    # legend layout (wrap when wider than the plot)
    legend_lines, lx, line = [], left, []
    for key, label, color in series:
        item_w = 24 + 9 * len(label) + 40
        if line and lx + item_w > w - right:
            legend_lines.append(line)
            line, lx = [], left
        line.append((lx, label, color))
        lx += item_w
    if line:
        legend_lines.append(line)
    rows_h = sum(row_hs)
    h = top + rows_h + 26 + 22 * len(legend_lines) + (22 if note else 0) + 20
    out = svg_open(w, h, title)
    ticks = 10
    for i in range(ticks + 1):
        v = xmin + (xmax - xmin) * i / ticks
        out.append(f"<line x1='{x(v):.1f}' y1='{top - 6}' x2='{x(v):.1f}' y2='{top + rows_h}' stroke='{GRID}' stroke-width='1'/>")
        out.append(f"<text x='{x(v):.1f}' y='{top - 12}' font-size='11' fill='{MUTED}' text-anchor='middle'>{v:.2f}</text>")
    y0 = top
    for ri, run in enumerate(runs):
        row_h, ls = row_hs[ri], labels[ri]
        bold = ' font-weight="600"' if run['alias'] in MIA else ''
        first_y = y0 + row_h / 2 - (len(ls) - 1) * line_h / 2 + 4
        for li, text in enumerate(ls):
            out.append(f"<text x='{left - 12}' y='{first_y + li * line_h:.1f}' font-size='13' fill='{INK}' text-anchor='end'{bold}>{esc(text)}</text>")
        bars_top = y0 + (row_h - (len(series) * (bar_h + gap) - gap)) / 2
        for si, (key, label, color) in enumerate(series):
            v = value_fn(run, key)
            y = bars_top + si * (bar_h + gap)
            out.append(f"<rect x='{left}' y='{y:.1f}' width='{max(0, x(v) - left):.1f}' height='{bar_h}' fill='{color}' rx='2'/>")
            out.append(f"<text x='{x(v) + 6:.1f}' y='{y + bar_h - 4:.1f}' font-size='11' fill='{INK}'>{fmt.format(v)}</text>")
        if ri:
            out.append(f"<line x1='24' y1='{y0}' x2='{w - 24}' y2='{y0}' stroke='{GRID}' stroke-width='0.5'/>")
        y0 += row_h
    ly = top + rows_h + 26
    for line in legend_lines:
        for lx, label, color in line:
            out.append(f"<rect x='{lx}' y='{ly - 11}' width='14' height='14' fill='{color}' rx='2'/>")
            out.append(f"<text x='{lx + 20}' y='{ly}' font-size='12' fill='{INK}'>{esc(label)}</text>")
        ly += 22
    if note:
        out.append(f"<text x='{left}' y='{ly}' font-size='11' fill='{MUTED}'>{esc(note)}</text>")
    out.append('</svg>')
    path.write_text('\n'.join(out), encoding='utf-8')


def confusion_chart(path, title, matrix, labels):
    n = len(labels)
    cell, left, top = 96, 150, 80
    w, h = left + n * cell + 60, top + n * cell + 60
    out = svg_open(w, h, title)
    out.append(f"<text x='{left + n * cell / 2:.1f}' y='{top - 30}' font-size='12' fill='{MUTED}' text-anchor='middle'>预测 →</text>")
    out.append(f"<text x='{left - 110}' y='{top + n * cell / 2:.1f}' font-size='12' fill='{MUTED}' transform='rotate(-90 {left - 110} {top + n * cell / 2:.1f})' text-anchor='middle'>真值 ↓</text>")
    for j, lab in enumerate(labels):
        out.append(f"<text x='{left + j * cell + cell / 2:.1f}' y='{top - 10}' font-size='13' fill='{INK}' text-anchor='middle'>{esc(lab)}</text>")
    for i, lab in enumerate(labels):
        row_sum = max(1, sum(matrix[i]))
        out.append(f"<text x='{left - 12}' y='{top + i * cell + cell / 2 + 5:.1f}' font-size='13' fill='{INK}' text-anchor='end'>{esc(lab)}</text>")
        for j in range(n):
            share = matrix[i][j] / row_sum
            alpha = 0.08 + 0.85 * share
            fill = f"rgba(9,105,218,{alpha:.3f})" if i == j else f"rgba(207,34,46,{alpha:.3f})"
            color = '#ffffff' if share > 0.5 else INK
            out.append(f"<rect x='{left + j * cell}' y='{top + i * cell}' width='{cell - 2}' height='{cell - 2}' fill='{fill}' rx='4'/>")
            out.append(f"<text x='{left + j * cell + cell / 2 - 1:.1f}' y='{top + i * cell + cell / 2 - 4:.1f}' font-size='18' font-weight='600' fill='{color}' text-anchor='middle'>{matrix[i][j]}</text>")
            out.append(f"<text x='{left + j * cell + cell / 2 - 1:.1f}' y='{top + i * cell + cell / 2 + 16:.1f}' font-size='11' fill='{color}' text-anchor='middle'>{share:.1%}</text>")
    out.append('</svg>')
    path.write_text('\n'.join(out), encoding='utf-8')


def line_chart(path, title, series, ylabel, ymin, ymax):
    """series: list of (label, color, [(x, y), ...]) with integer x."""
    w, h, left, right, top, bottom = 760, 400, 70, 200, 56, 50
    out = svg_open(w, h, title)
    xs = sorted({x for _, _, pts in series for x, _ in pts})
    plot_w, plot_h = w - left - right, h - top - bottom
    def X(x):
        return left + (xs.index(x)) / max(1, len(xs) - 1) * plot_w
    def Y(y):
        return top + plot_h - (y - ymin) / (ymax - ymin) * plot_h
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        out.append(f"<line x1='{left}' y1='{Y(v):.1f}' x2='{w - right}' y2='{Y(v):.1f}' stroke='{GRID}'/>")
        out.append(f"<text x='{left - 8}' y='{Y(v) + 4:.1f}' font-size='11' fill='{MUTED}' text-anchor='end'>{v:.2f}</text>")
    for x in xs:
        out.append(f"<text x='{X(x):.1f}' y='{h - bottom + 20}' font-size='12' fill='{MUTED}' text-anchor='middle'>epoch {x}</text>")
    out.append(f"<text x='{left - 50}' y='{top - 14}' font-size='11' fill='{MUTED}'>{esc(ylabel)}</text>")
    for si, (label, color, pts) in enumerate(series):
        d = ' '.join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
        out.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        for x, y in pts:
            out.append(f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='4' fill='{color}'/>")
            out.append(f"<text x='{X(x):.1f}' y='{Y(y) - 9:.1f}' font-size='10' fill='{INK}' text-anchor='middle'>{y:.4f}</text>")
        ly = top + 10 + si * 22
        out.append(f"<rect x='{w - right + 16}' y='{ly - 10}' width='14' height='14' fill='{color}' rx='2'/>")
        out.append(f"<text x='{w - right + 36}' y='{ly + 1}' font-size='12' fill='{INK}'>{esc(label)}</text>")
    out.append('</svg>')
    path.write_text('\n'.join(out), encoding='utf-8')


def stacked_chart(path, title, rows):
    """rows: list of (label, {class: count})."""
    w, h, left, right, top, row_h = 880, 60 + 64 * len(rows) + 50, 250, 40, 56, 64
    out = svg_open(w, h, title)
    plot_w = w - left - right
    for ri, (label, counts) in enumerate(rows):
        total = sum(counts.values())
        y = top + ri * row_h
        out.append(f"<text x='{left - 12}' y='{y + 26}' font-size='13' fill='{INK}' text-anchor='end'>{esc(label)}</text>")
        out.append(f"<text x='{left - 12}' y='{y + 44}' font-size='11' fill='{MUTED}' text-anchor='end'>共 {total:,} 行</text>")
        x = left
        for lab in LABELS:
            share = counts[lab] / total
            seg = share * plot_w
            out.append(f"<rect x='{x:.1f}' y='{y + 8}' width='{seg:.1f}' height='36' fill='{CLASS_COLORS[lab]}' opacity='0.85'/>")
            if seg > 60:
                out.append(f"<text x='{x + seg / 2:.1f}' y='{y + 31}' font-size='12' fill='#ffffff' text-anchor='middle'>{lab} {counts[lab]:,}（{share:.0%}）</text>")
            x += seg
    out.append('</svg>')
    path.write_text('\n'.join(out), encoding='utf-8')


# 1. accuracy / macro-F1 / neutral recall across all runs and sets
hbar_chart(CHARTS / 'accuracy.svg', '准确率：原版零样本基线与全部 10 次训练在统一评测集上', ALL,
           lambda r, k: metric(r, k, 'accuracy'), SETS, 0.4, 1.0, note='所有模型用同一脚本、同一评测集重评')
hbar_chart(CHARTS / 'macro-f1.svg', 'Macro-F1：原版零样本基线与全部 10 次训练在统一评测集上', ALL,
           lambda r, k: metric(r, k, 'macro_f1'), SETS, 0.4, 1.0)
hbar_chart(CHARTS / 'neutral-recall.svg', '中性召回：最难的一类', ALL,
           lambda r, k: metric(r, k, 'neutral_recall'), SETS, 0.3, 1.0)

# 2. per-class recall on holdout for the four key runs
KEY = [BY['laya-oldlabels'], BY['mia-laya'], BY['qwen2b-relabeled'], BY['mia-qwen3.5-2b']]
hbar_chart(CHARTS / 'per-class-recall-holdout.svg', f'holdout {_rows("holdout")} 上各类别召回：四个关键版本', KEY,
           lambda r, k: r['unified_holdout']['per_class'][k]['recall'],
           [(lab, f'{lab}召回', CLASS_COLORS[lab]) for lab in LABELS], 0.3, 1.0)
hbar_chart(CHARTS / 'per-class-f1-holdout.svg', f'holdout {_rows("holdout")} 上各类别 F1：四个关键版本', KEY,
           lambda r, k: r['unified_holdout']['per_class'][k]['f1'],
           [(lab, f'{lab} F1', CLASS_COLORS[lab]) for lab in LABELS], 0.3, 1.0)

# 3. confusion matrices for the two released versions
for alias in ('mia-laya', 'mia-qwen3.5-2b'):
    confusion_chart(CHARTS / f'confusion-{alias}-holdout.svg', f'{alias}：holdout {_rows("holdout")} 混淆矩阵', BY[alias]['unified_holdout']['confusion_matrix'], LABELS)
    confusion_chart(CHARTS / f'confusion-{alias}-csv.svg', f'{alias}：中文 900 条混淆矩阵', BY[alias]['chinese_csv']['pooled']['confusion_matrix'], LABELS)

# 4. epoch curves for the Qwen runs (dev macro-F1 recorded per epoch)
qwen_series = []
for alias, color in (('qwen0.8b-relabeled', '#bf8700'), ('qwen2b-relabeled', '#8250df'), ('mia-qwen3.5-2b', '#0969da')):
    pts = [(h['epoch'], h['dev_macro_f1']) for h in BY[alias]['history'] if 'dev_macro_f1' in h]
    qwen_series.append((alias, color, pts))
line_chart(CHARTS / 'epochs-qwen.svg', 'Qwen 三次训练每个 epoch 结束时的开发集 Macro-F1', qwen_series, 'Macro-F1（各自训练时的开发集）', 0.80, 0.88)
loss_series = []
palette = ['#cf222e', '#bf8700', '#1a7f37', '#0969da', '#8250df', '#6e7781', '#d4a72c', '#0550ae', '#a40e26', '#116329']
for i, r in enumerate(RUNS):
    loss_series.append((r['alias'], palette[i % len(palette)], [(h['epoch'], h['train_loss']) for h in r['history']]))
line_chart(CHARTS / 'train-loss.svg', '每个 epoch 的平均训练损失（全部 10 次训练）', loss_series, '交叉熵', 0.0, 0.7)

# 5. label distribution: old machine labels vs relabeled (train split), from README-recorded counts
stacked_chart(CHARTS / 'label-distribution.svg', '训练集标签分布：旧机器标签 → 大模型重标 → 褒贬混合并入中性', [
    ('早期机器标签', {'负面': 18657, '中性': 20728, '正面': 37874}),
    ('大模型重标，褒贬混合剔除', {'负面': 14525, '中性': 30858, '正面': 27227}),
    ('大模型重标，褒贬混合并入中性', {'负面': 14525, '中性': 34360, '正面': 27227}),
])

# 5b. public benchmarks: forced two-way accuracy on the binary sets, three-way accuracy on DMSC
PUB = SUMMARY.get('public_benchmarks', {})
PUB_MODELS = [a for a in ('mia-qwen3.5-2b', 'mia-laya', 'laya-oldlabels', 'qwen3.5-2b-zero-shot') if a in PUB.get('results', {})]
if PUB_MODELS:
    pub_rows = []
    for s in PUB['sets']:
        pub_rows.append({'alias': s['name'], 'key': s['key'], 'kind': s['kind']})
    def pub_value(row, alias):
        r = PUB['results'][alias][row['key']]
        return r['forced_binary_accuracy'] if row['kind'] == 'binary' else r['accuracy']
    hbar_chart(CHARTS / 'public-benchmarks.svg', '公开数据集：二分类集用二选一准确率，DMSC 用三分类准确率', pub_rows, pub_value,
               [(a, a, c) for a, c in zip(PUB_MODELS, ('#0969da', '#8250df', '#bf8700', '#6e7781'))], 0.4, 1.0,
               note='模型没有在这些数据集上训练过')

# 6. the journey: holdout accuracy along the chronological path
journey = [BY[a] for a in ('qwen3.5-2b-zero-shot', 'laya-oldlabels', 'laya-oldlabels-cw-balanced', 'mia-laya', 'qwen2b-relabeled', 'mia-qwen3.5-2b') if a in BY]
hbar_chart(CHARTS / 'journey.svg', '一路走来：holdout 准确率随每一步的变化', journey,
           lambda r, k: metric(r, k, 'accuracy'), [('unified_holdout', 'holdout 准确率', '#0969da')], 0.6, 0.9,
           note='原版零样本 → 起点 → 类别权重 → 换标签 → 换底座 → 定义褒贬混合')

# ---- index.html --------------------------------------------------------------------------------
def tri(m):
    return f"{m['accuracy']:.4f} / {m['macro_f1']:.4f} / {m['per_class']['中性']['recall']:.3f}"


def table(headers, rows):
    head = ''.join(f'<th>{esc(h)}</th>' for h in headers)
    body = '\n'.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>\n{body}\n</tbody></table>"


def name_cell(r):
    tag = ' <span class="tag">发布版</span>' if r['alias'] in MIA else ''
    return f"<code>{esc(r['alias'])}</code>{tag}"


overview_rows = [[name_cell(r), esc(r['base']), f"{r['params_million']}M" if 'params_million' in r else esc(r.get('params', '')),
                  esc(r['labels']), r['class_weights'], r['epochs'],
                  tri(r['unified_dev']), tri(r['unified_holdout']), tri(r['chinese_csv']['pooled'])] for r in ALL]
per_class_rows = []
for r in ALL:
    for name, m in (('开发集', r['unified_dev']), ('holdout', r['unified_holdout']), ('中文 CSV', r['chinese_csv']['pooled'])):
        per_class_rows.append([name_cell(r), name] + [f"{m['per_class'][k]['precision']:.3f} / {m['per_class'][k]['recall']:.3f} / {m['per_class'][k]['f1']:.3f}" for k in LABELS] + [esc(m['confusion_matrix'])])
csv_rows = []
for r in ALL:
    for fname, m in r['chinese_csv']['files'].items():
        overlap = m.get('train_text_overlap')
        csv_rows.append([name_cell(r), f'<code>{esc(fname)}</code>', f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}", f"{m['per_class']['中性']['recall']:.3f}", '—' if overlap is None else overlap])
public_rows = []
for s in PUB.get('sets', []):
    res = {a: PUB['results'][a][s['key']] for a in PUB_MODELS}
    n = f"{res[PUB_MODELS[0]]['rows']:,}" if PUB_MODELS else ''
    if s['kind'] == 'binary':
        public_rows.append([esc(s['name']), n, '二选一准确率（只比较正/负概率）'] + [f"{res[a]['forced_binary_accuracy']:.4f}" for a in PUB_MODELS])
        public_rows.append(['', '', '三选一严格准确率（判中性算错）'] + [f"{res[a]['accuracy']:.4f}" for a in PUB_MODELS])
        public_rows.append(['', '', '判为中性的比例'] + [f"{res[a]['neutral_rate']:.3f}" for a in PUB_MODELS])
        public_rows.append(['', '', '只看给出极性的行的准确率'] + [f"{res[a]['polarity_accuracy']:.4f}" for a in PUB_MODELS])
    else:
        public_rows.append([esc(s['name']), n, '三分类准确率'] + [f"{res[a]['accuracy']:.4f}" for a in PUB_MODELS])
        public_rows.append(['', '', '三分类 Macro-F1'] + [f"{res[a]['macro_f1']:.4f}" for a in PUB_MODELS])
        public_rows.append(['', '', '中性（3 星）召回'] + [f"{res[a]['per_class']['中性']['recall']:.3f}" for a in PUB_MODELS])
train_rows = [[name_cell(r), esc(r['train_data']), f"{r['train_rows']:,}", r['own_dev']['rows'], f"{r['own_dev']['accuracy']:.4f}", f"{r['own_dev']['macro_f1']:.4f}",
               f"{r['temperature']:.4f}", f"{r['own_dev']['ece_before']:.4f}", f"{r['own_dev']['ece_after']:.4f}", esc([round(h['train_loss'], 4) for h in r['history']])] for r in RUNS]

charts = [
    ('journey.svg', '一路走来', '五个关键版本在同一 holdout 上的准确率。前两步是配方，后三步分别是换标签、换底座、把褒贬混合定义清楚。'),
    ('accuracy.svg', '准确率', '10 次训练全部用同一脚本在三套评测集上重评。'),
    ('macro-f1.svg', 'Macro-F1', '三类 F1 的算术平均，对少数类更敏感。'),
    ('neutral-recall.svg', '中性召回', '中性一直是最难的一类；旧标签模型的中性召回不到 0.55。'),
    ('per-class-recall-holdout.svg', '各类别召回', '换标签后不再把客观陈述判成正面；换底座后负面与中性同时上来。'),
    ('per-class-f1-holdout.svg', '各类别 F1', '同一 holdout 上的各类别 F1。'),
    ('confusion-mia-qwen3.5-2b-holdout.svg', 'Mia-Qwen3.5-2B 混淆矩阵（holdout）', '行是真值，列是预测，括号内是行内占比。剩余错误集中在负面↔中性。'),
    ('confusion-mia-laya-holdout.svg', 'Mia-Laya 混淆矩阵（holdout）', '与上图同一评测集。'),
    ('confusion-mia-qwen3.5-2b-csv.svg', 'Mia-Qwen3.5-2B 混淆矩阵（中文 900 条）', '正面与负面接近全对，剩余错误是中性判负面、正面判中性。'),
    ('confusion-mia-laya-csv.svg', 'Mia-Laya 混淆矩阵（中文 900 条）', '同一 900 条。'),
    ('label-distribution.svg', '标签分布', '重标把大量客观陈述从正面/负面挪回中性；中性从 27% 升到 45%。'),
    ('public-benchmarks.svg', '公开数据集', '两个发布版和起点模型都没有在这些集上训练过。二分类集只比较正、负两类概率；DMSC 用 3 星当中性，只是近似。'),
    ('epochs-qwen.svg', 'Qwen 每个 epoch 的开发集 Macro-F1', '第 2 个 epoch 到顶，第 3 个 epoch 只是在记训练集。'),
    ('train-loss.svg', '训练损失', 'Qwen 系第 3 个 epoch 损失降到 0.02，属于过拟合；Laya 系降得慢。'),
]
chart_html = '\n'.join(f"<figure><img src='charts/{f}' alt='{esc(t)}' loading='lazy'/><figcaption><strong>{esc(t)}</strong> {esc(c)}</figcaption></figure>" for f, t, c in charts)

page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Mia · 中文情感三分类小模型</title>
<style>
  :root {{ --ink:{INK}; --muted:{MUTED}; --line:{GRID}; --bg:#ffffff; --soft:#f6f8fa; --accent:#0969da; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }}
  main {{ max-width: 1120px; margin: 0 auto; padding: 32px 16px 80px; }}
  h1 {{ font-size: 32px; margin: 0 0 8px; }}
  h2 {{ font-size: 22px; margin: 48px 0 12px; padding-top: 16px; border-top: 1px solid var(--line); }}
  p.lead {{ color: var(--muted); font-size: 17px; margin: 0 0 24px; }}
  .cards {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 24px 0; }}
  .card {{ border:1px solid var(--line); border-radius: 10px; padding: 16px 18px; background: var(--soft); }}
  .card h3 {{ margin: 0 0 6px; font-size: 18px; }}
  .card .big {{ font-size: 28px; font-weight: 700; }}
  .card .sub {{ color: var(--muted); font-size: 13px; }}
  figure {{ margin: 24px 0; }}
  figure img {{ width: 100%; height: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
  figcaption {{ color: var(--muted); font-size: 14px; margin-top: 8px; }}
  .tablewrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; margin: 16px 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; vertical-align: top; }}
  th {{ background: var(--soft); position: sticky; top: 0; }}
  code {{ background: var(--soft); padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }}
  .tag {{ display:inline-block; background: var(--accent); color:#fff; font-size: 11px; padding: 1px 6px; border-radius: 10px; margin-left: 6px; vertical-align: middle; }}
  a {{ color: var(--accent); }}
  .note {{ color: var(--muted); font-size: 14px; }}
</style>
</head>
<body>
<main>
<h1>Mia</h1>
<p class="lead">中文短文本情感三分类（负面 / 中性 / 正面）小模型。两个发布版本：<code>mia-laya</code>（322M，Laya 判别头）与 <code>mia-qwen3.5-2b</code>（1.9B，Qwen3.5-2B 文本主干直接做分类）。这一页只放数字和图；来龙去脉、数据、训练心得在 <a href="../README.md">README</a>。</p>

<div class="cards">
  <div class="card"><h3>mia-qwen3.5-2b</h3><div class="big">{BY['mia-qwen3.5-2b']['unified_holdout']['accuracy']:.4f}</div><div class="sub">holdout 准确率 · Macro-F1 {BY['mia-qwen3.5-2b']['unified_holdout']['macro_f1']:.4f} · 中文 900 条 {BY['mia-qwen3.5-2b']['chinese_csv']['pooled']['accuracy']:.4f}</div></div>
  <div class="card"><h3>mia-laya</h3><div class="big">{BY['mia-laya']['unified_holdout']['accuracy']:.4f}</div><div class="sub">holdout 准确率 · Macro-F1 {BY['mia-laya']['unified_holdout']['macro_f1']:.4f} · 中文 900 条 {BY['mia-laya']['chinese_csv']['pooled']['accuracy']:.4f}</div></div>
  <div class="card"><h3>起点 laya-oldlabels</h3><div class="big">{BY['laya-oldlabels']['unified_holdout']['accuracy']:.4f}</div><div class="sub">同一 holdout 上的早期标签模型 · 从这里到发布版靠的是换标签、换底座、定义褒贬混合</div></div>
  {"<div class='card'><h3>原版 Qwen3.5-2B 零样本</h3><div class='big'>" + f"{ZS['unified_holdout']['accuracy']:.4f}" + "</div><div class='sub'>未微调，同一口径提示词、受限选择 · 中文 900 条 " + f"{ZS['chinese_csv']['pooled']['accuracy']:.4f}" + "</div></div>" if ZS else ""}
</div>

<h2>图表</h2>
{chart_html}

<h2>全部训练在统一评测集上的成绩</h2>
<p class="note">评测集：开发集 {esc(SUMMARY['unified_eval']['dev'])}；holdout {esc(SUMMARY['unified_eval']['holdout'])}；中文 CSV {esc(SUMMARY['unified_eval']['chinese_csv'])}。开发集与 holdout 的标签来自大模型重标（褒贬混合并入中性），中文 CSV 是独立标注的模板化基准。每格为 准确率 / Macro-F1 / 中性召回。</p>
<div class="tablewrap">{table(['模型', '底座', '参数量', '训练标签', '类别权重', 'epochs', f'开发集 {_rows("dev")}', f'holdout {_rows("holdout")}', '中文 CSV 900'], overview_rows)}</div>

<h2>各类别精确率 / 召回 / F1 与混淆矩阵</h2>
<div class="tablewrap">{table(['模型', '集合', '负面 P / R / F1', '中性 P / R / F1', '正面 P / R / F1', '混淆矩阵（行=真值 负/中/正）'], per_class_rows)}</div>

<h2>中文 900 条分表</h2>
<div class="tablewrap">{table(['模型', 'CSV', 'Acc', 'Macro-F1', '中性召回', '与训练文本精确重叠'], csv_rows)}</div>

<h2>公开数据集</h2>
<p class="note">模型没有在这些数据集上训练过。二分类集给四个数：二选一准确率只比较正、负两类概率；三选一严格准确率把判中性算错；判为中性的比例；只看给出极性的行的准确率。DMSC 用 1-2 星为负、3 星为中、4-5 星为正，3 星当中性只是近似。文本做与训练相同的归一化并截到 512 字。</p>
<div class="tablewrap">{table(['公开数据集', '行数', '指标'] + PUB_MODELS, public_rows)}</div>

<h2>训练侧：各自训练时的开发集、温度与校准</h2>
<p class="note">每次训练在自己那套数据的开发集上评一次；旧标签模型的开发集是旧标签，所以这一列不能跨行比较。温度在 calibration 切分上拟合；ECE 是 15 桶期望校准误差。</p>
<div class="tablewrap">{table(['模型', '训练数据', '训练行数', '自有开发集行数', 'Acc', 'Macro-F1', '温度', '校准前 ECE', '校准后 ECE', '每 epoch 训练损失'], train_rows)}</div>

<p class="note">数据来源：<code>reports/summary.json</code>，由 <code>scripts</code> 目录外的 <code>build_docs.py</code> 生成本页与全部 SVG。</p>
</main>
</body>
</html>
"""
(ROOT / 'docs' / 'index.html').write_text(page, encoding='utf-8')
print('WROTE', ROOT / 'docs' / 'index.html', 'and', len(list(CHARTS.glob('*.svg'))), 'charts')

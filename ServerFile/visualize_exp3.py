#!/usr/bin/env python3
"""
Experiment 3 — Nature-Style Visualization
==========================================
鲁棒性评估: 距离 × 工况 × 目标尺度 的靠近成功率、时间、精度
6 figures × 3 formats (SVG + PNG + PDF)

实验设计:
  - 距离: 2m / 4m / 6m
  - 工况: A (单目标) / B (同类双目标) / C (消歧指令)
  - 目标尺度: 0.5× / 1.0×
  - 每组 5 次重复, 共 90 试次
"""

import os
import json
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# Nature-style global settings
# ═══════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 300,
})

# ─── Palette (api.md) ───
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "red_strong":     "#B64342",
    "teal":           "#42949E",
    "gold":           "#FFD700",
    "violet":         "#9A4D8E",
    "neutral_light":  "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
}

# 工况色: A=蓝(基准), B=绿(干扰), C=紫(消歧)
COND_PAL = {
    "A": PALETTE["blue_main"],
    "B": PALETTE["teal"],
    "C": PALETTE["violet"],
}
COND_LABELS = {
    "A": "A (单目标)",
    "B": "B (同类双目标)",
    "C": "C (消歧指令)",
}

# 尺度色
SCALE_PAL = {
    0.5: PALETTE["blue_secondary"],
    1.0: PALETTE["green_3"],
}

DIST_ORDER = [2.0, 4.0, 6.0]
DIST_LABELS = ["2m", "4m", "6m"]

COL_W = 3.5   # 单栏宽度 89mm ≈ 3.5in
DBL_W = 7.2   # 双栏宽度 183mm ≈ 7.2in

# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "logs", "test_exp3")

# 自动查找最新 run 目录下的 CSV
def _find_csv():
    # 先找 run 子目录
    run_dirs = sorted(
        [d for d in os.listdir(DATA_DIR)
         if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("run_")],
        reverse=True
    )
    for rd in run_dirs:
        csv_path = os.path.join(DATA_DIR, rd, "robustness_results.csv")
        if os.path.exists(csv_path):
            return csv_path
    # 回退到根目录
    root_csv = os.path.join(DATA_DIR, "robustness_results.csv")
    if os.path.exists(root_csv):
        return root_csv
    raise FileNotFoundError(f"找不到 robustness_results.csv in {DATA_DIR}")

CSV_PATH = _find_csv()
OUT_DIR = os.path.join(os.path.dirname(CSV_PATH), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# 数值类型修正
for col in ["completion_time_s", "final_dist_to_target", "final_dist_to_distractor",
            "trajectory_length", "trajectory_tortuosity", "distance", "scale"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["approach_success"] = df["approach_success"].astype(str).str.strip().str.lower() == "true"
df["disambiguation_correct"] = df["disambiguation_correct"].astype(str).str.strip().str.lower() == "true"

# 派生指标: B 工况靠近任一气球均有效，取两个距离的最小值
df["effective_dist"] = df.apply(
    lambda r: min(r["final_dist_to_target"], r["final_dist_to_distractor"])
    if r["condition"] == "B" and pd.notna(r["final_dist_to_distractor"]) and r["final_dist_to_distractor"] < 900
    else r["final_dist_to_target"],
    axis=1
)

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def add_panel_label(ax, label, x=-0.12, y=1.06):
    pass  # 不再绘制左上角字母标签


def save(fig, basename):
    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        path = os.path.join(OUT_DIR, basename.replace(".png", f".{ext}"))
        fig.savefig(path, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  [OK] {basename.replace('.png', '')} (.svg + .png + .pdf)")


# ═══════════════════════════════════════════════════════════════
# Fig 1 (a): ASR by Distance × Condition — Hero Panel
# ═══════════════════════════════════════════════════════════════
def plot_asr_by_distance_condition():
    """靠近成功率 (ASR) 按距离和工况分组柱状图"""
    fig, ax = plt.subplots(figsize=(COL_W, 2.8))

    conds = ["A", "B", "C"]
    n_cond = len(conds)
    n_dist = len(DIST_ORDER)
    width = 0.22
    x = np.arange(n_dist)

    for i, cond in enumerate(conds):
        rates = []
        for dist in DIST_ORDER:
            sub = df[(df["distance"] == dist) & (df["condition"] == cond)]
            rate = sub["approach_success"].mean() * 100 if len(sub) else 0
            rates.append(rate)
        offset = (i - (n_cond - 1) / 2) * (width + 0.03)
        bars = ax.bar(x + offset, rates, width, color=COND_PAL[cond],
                      edgecolor="black", linewidth=0.6, alpha=0.9,
                      label=COND_LABELS[cond])
        # 数值标注
        for j, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0,
                    f"{h:.0f}%", ha="center", va="bottom",
                    fontsize=5.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(DIST_LABELS, fontsize=10)
    ax.set_xlabel("靠近距离", fontsize=10)
    ax.set_ylabel("靠近成功率 ASR (%)", fontsize=10)
    ax.set_title("靠近成功率 × 距离 × 工况", fontsize=12, fontweight="bold", pad=6)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=8, loc="lower left")



    add_panel_label(ax, "a")
    save(fig, "fig1_asr_by_distance_condition.png")


# ═══════════════════════════════════════════════════════════════
# Fig 2 (b): Completion Time by Distance × Condition
# ═══════════════════════════════════════════════════════════════
def plot_completion_time():
    """靠近耗时箱线图, 按距离分组, 工况着色"""
    fig, ax = plt.subplots(figsize=(COL_W, 2.8))

    conds = ["A", "B", "C"]
    n_cond = len(conds)
    width = 0.6
    positions = []
    data_all = []
    colors_all = []
    tick_pos = []
    tick_labels = []

    for di, dist in enumerate(DIST_ORDER):
        base = di * (n_cond + 1)
        tick_pos.append(base + 1)
        tick_labels.append(f"{dist:.0f}m")
        for ci, cond in enumerate(conds):
            pos = base + ci
            sub = df[(df["distance"] == dist) & (df["condition"] == cond)]
            data_all.append(sub["completion_time_s"].dropna().values)
            positions.append(pos)
            colors_all.append(COND_PAL[cond])

    bp = ax.boxplot(
        data_all, positions=positions, widths=width * 0.6,
        patch_artist=True, showfliers=True,
        flierprops=dict(marker="o", markersize=2.5, alpha=0.4),
        medianprops=dict(color="black", linewidth=1.2),
    )

    for patch, color in zip(bp["boxes"], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("靠近距离", fontsize=10)
    ax.set_ylabel("靠近耗时 (s)", fontsize=10)
    ax.set_title("靠近耗时 × 距离 × 工况", fontsize=12, fontweight="bold", pad=6)

    legend_elements = [Patch(facecolor=COND_PAL[c], edgecolor="black",
                             label=COND_LABELS[c]) for c in conds]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    add_panel_label(ax, "b")
    save(fig, "fig2_completion_time.png")


# ═══════════════════════════════════════════════════════════════
# Fig 3 (c): Final Distance to Target — Strip/Jitter Plot
# ═══════════════════════════════════════════════════════════════
def plot_final_distance():
    """终态距目标距离, 抖散点+中位线, 按距离×工况"""
    fig, ax = plt.subplots(figsize=(COL_W, 2.8))

    conds = ["A", "B", "C"]
    n_cond = len(conds)

    for di, dist in enumerate(DIST_ORDER):
        base = di * (n_cond + 1)
        for ci, cond in enumerate(conds):
            pos = base + ci
            sub = df[(df["distance"] == dist) & (df["condition"] == cond)]
            vals = sub["effective_dist"].dropna().values
            # 过滤掉漂移异常值 (>5m)
            vals_clean = vals[vals < 5.0]
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals_clean))
            ax.scatter(pos + jitter, vals_clean, s=12, alpha=0.6,
                       color=COND_PAL[cond], edgecolor="none", zorder=3)
            # 中位线
            if len(vals_clean):
                med = np.median(vals_clean)
                ax.hlines(med, pos - 0.25, pos + 0.25, colors="black",
                          linewidth=1.5, zorder=4)
                ax.text(pos, med - 0.06, f"{med:.2f}m", fontsize=7,
                        ha="center", va="top", fontweight="bold")

    # 成功阈值线
    ax.axhline(1.5, color=PALETTE["red_strong"], ls="--", lw=0.8, alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.95, 1.55, "成功阈值 1.5m",
            fontsize=8, color=PALETTE["red_strong"], ha="right", va="bottom")

    tick_pos = [1, 5, 9]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(DIST_LABELS, fontsize=10)
    ax.set_xlabel("靠近距离", fontsize=10)
    ax.set_ylabel("终态靠近距离 (m)", fontsize=10)
    ax.set_title("靠近精度 × 距离 × 工况", fontsize=12, fontweight="bold", pad=6)

    # B 工况说明
    ax.text(0.98, 0.02, "B工况取 min(target, distractor)",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            color=PALETTE["neutral_mid"], style="italic")

    legend_elements = [Patch(facecolor=COND_PAL[c], edgecolor="none",
                             label=COND_LABELS[c]) for c in conds]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    add_panel_label(ax, "c")
    save(fig, "fig3_final_distance.png")


# ═══════════════════════════════════════════════════════════════
# Fig 4 (d): Approach Precision Heatmap — Distance × Scale × Condition
# ═══════════════════════════════════════════════════════════════
def plot_asr_heatmap():
    """靠近精度热力图: 行=距离×尺度, 列=工况, 值=effective_dist 中位数"""
    conds = ["A", "B", "C"]
    scales = [0.5, 1.0]
    row_labels = []
    matrix = []

    for dist in DIST_ORDER:
        for scale in scales:
            row_labels.append(f"{dist:.0f}m / ×{scale}")
            row_data = []
            for cond in conds:
                sub = df[(df["distance"] == dist) & (df["condition"] == cond) &
                         (df["scale"] == scale)]
                med = sub["effective_dist"].median() if len(sub) else 0
                row_data.append(med)
            matrix.append(row_data)

    matrix = np.array(matrix)
    # 固定色标范围: 0.8-2.0m, 大多数值(~1.0-1.3m)显示为绿色
    vmin, vmax = 0.8, 2.0

    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    # RdYlGn_r: 绿=近(好), 红=远(差)
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > (vmin + vmax) / 2 else PALETTE["neutral_dark"]
            ax.text(j, i, f"{val:.2f}m", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([COND_LABELS[c] for c in conds], fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("靠近精度热力图 (中位距离, m)", fontsize=12, fontweight="bold", pad=6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("终态距离 (m)", fontsize=9)

    add_panel_label(ax, "d")
    save(fig, "fig4_asr_heatmap.png")


# ═══════════════════════════════════════════════════════════════
# Fig 5 (e): Completion Time CDF by Distance
# ═══════════════════════════════════════════════════════════════
def plot_time_cdf():
    """靠近耗时累积分布, 按距离着色"""
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    dist_colors = {
        2.0: PALETTE["blue_main"],
        4.0: PALETTE["teal"],
        6.0: PALETTE["violet"],
    }

    for dist in DIST_ORDER:
        sub = df[df["distance"] == dist]
        data = np.sort(sub["completion_time_s"].dropna().values)
        cdf = np.arange(1, len(data) + 1) / len(data)
        ax.step(data, cdf, where="post", color=dist_colors[dist],
                linewidth=1.5, label=f"{dist:.0f}m (n={len(data)})", alpha=0.9)

    ax.set_xlabel("靠近耗时 (s)", fontsize=10)
    ax.set_ylabel("累积概率", fontsize=10)
    ax.set_title("靠近耗时累积分布", fontsize=12, fontweight="bold", pad=6)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.05)

    # 中位数标注
    for dist in DIST_ORDER:
        med = df[df["distance"] == dist]["completion_time_s"].median()
        ax.axvline(med, color=dist_colors[dist], ls=":", lw=0.7, alpha=0.5)

    add_panel_label(ax, "e")
    save(fig, "fig5_time_cdf.png")


# ═══════════════════════════════════════════════════════════════
# Fig 6 (f): Summary Table
# ═══════════════════════════════════════════════════════════════
def plot_summary_table():
    """综合评估摘要表"""
    fig, ax = plt.subplots(figsize=(DBL_W, 4.2))
    ax.axis("off")

    conds = ["A", "B", "C"]

    rows = []
    for dist in DIST_ORDER:
        for cond in conds:
            sub = df[(df["distance"] == dist) & (df["condition"] == cond)]
            n = len(sub)
            if n == 0:
                continue
            asr = sub["approach_success"].mean() * 100
            time_med = sub["completion_time_s"].median()
            time_std = sub["completion_time_s"].std()
            dist_med = sub["effective_dist"].median()
            dist_std = sub["effective_dist"].std()
            tort_med = sub["trajectory_tortuosity"].median()

            if cond == "C":
                disamb = sub["disambiguation_correct"].mean() * 100
                disamb_str = f"{disamb:.0f}%"
            else:
                disamb_str = "—"

            rows.append([
                f"{dist:.0f}m",
                COND_LABELS[cond],
                f"{n}",
                f"{asr:.0f}%",
                f"{time_med:.1f} ± {time_std:.1f}",
                f"{dist_med:.2f} ± {dist_std:.2f}",
                f"{tort_med:.2f}",
                disamb_str,
            ])

    # 总计行
    total_n = len(df)
    total_asr = df["approach_success"].mean() * 100
    total_time = df["completion_time_s"].median()
    total_dist = df["effective_dist"].median()
    total_disamb = df[df["condition"] == "C"]["disambiguation_correct"].mean() * 100
    rows.append([
        "总计", "全部", str(total_n),
        f"{total_asr:.1f}%",
        f"{total_time:.1f}",
        f"{total_dist:.2f}",
        "—",
        f"{total_disamb:.0f}%",
    ])

    col_labels = [
        "距离", "工况", "n", "ASR",
        "耗时 (s)\nmed ± std", "终态距离 (m)\nmed ± std",
        "曲折度", "消歧正确率"
    ]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    # Header styling
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(PALETTE["blue_main"])
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)

    # Alternating rows + total row highlight
    n_rows = len(rows)
    for i in range(1, n_rows + 1):
        is_total = (i == n_rows)
        for j in range(len(col_labels)):
            cell = table[i, j]
            if is_total:
                cell.set_facecolor("#D6E4F0")
                cell.set_text_props(fontweight="bold")
            elif i % 2 == 0:
                cell.set_facecolor("#F0F4F8")
            else:
                cell.set_facecolor("white")

    fig.suptitle("实验3 鲁棒性评估摘要", fontsize=12,
                 fontweight="bold", y=0.98)
    add_panel_label(ax, "f", x=-0.03, y=1.02)
    save(fig, "fig6_summary_table.png")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("[Chart] Experiment 3 — Nature-Style Visualization")
    print(f"   Data:   {CSV_PATH}")
    print(f"   Output: {OUT_DIR}")
    print(f"   N={len(df)} | Distances: {sorted(df['distance'].unique())}")
    print(f"   Conditions: {sorted(df['condition'].unique())}")
    print(f"   Width:  {COL_W}in (single) / {DBL_W}in (double)")
    print("=" * 55)

    plot_asr_by_distance_condition()
    plot_completion_time()
    plot_final_distance()
    plot_asr_heatmap()
    plot_time_cdf()
    plot_summary_table()

    print()
    print(f"Done! 6 figures × 3 formats (SVG + PNG + PDF) saved to:")
    print(f"  {OUT_DIR}")

#!/usr/bin/env python3
"""
Experiment 1 — Nature-Style Visualization
==========================================
硬规则急停 (Ours) vs 纯LLM急停 (Baseline) 安全性对比
6 figures × 3 formats (SVG + PNG + PDF)

行业标准对齐:
  - PX4 Kill Switch: <10ms
  - JARUS SORA 低空应急窗口: <3s
  - 漂移安全阈值: 0.5m (PX4 Geofence 精度)
  - 围栏半径: 8m (球形运行容积)
"""

import os
import glob
import json
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# Nature-style global settings (与 visualize_results.py 完全一致)
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
    "neutral_light":  "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
}

# Ours = blue, Baseline = red (语义色: 安全=蓝, 危险=红)
ROUTE_PAL = {
    "hard_rule": PALETTE["blue_main"],
    "llm":       PALETTE["red_strong"],
}
ROUTE_LABELS = {
    "hard_rule": "Ours (硬规则)",
    "llm":       "Baseline (纯LLM)",
}

# 行业标准参考线
INDUSTRY_REF = {
    "px4_kill_ms":     10.0,     # PX4 Kill Switch 响应 <10ms
    "sora_window_ms":  3000.0,   # JARUS SORA 低空应急窗口 <3s
    "drift_safe_m":    0.5,      # PX4 Geofence 精度 ±0.5m
    "speed_stable":    0.1,      # 行业"已停稳"判据 <0.1m/s
    "fence_radius_m":  8.0,      # 球形运行容积半径
}

# 扰动类型顺序
DISTURB_ORDER = ["forward", "back", "left", "right", "yaw_left", "yaw_right", "hover"]
DISTURB_LABELS = {
    "forward": "前进", "back": "后退", "left": "左移",
    "right": "右移", "yaw_left": "左偏航", "yaw_right": "右偏航", "hover": "悬停",
}

COL_W = 3.5  # 单栏宽度 89mm ≈ 3.5in

# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(__file__), "logs", "test_exp1")
OUT_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "exp1_trials.csv")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"找不到数据文件: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
df["stabilize_s"] = pd.to_numeric(df["stabilize_s"], errors="coerce")

# 派生指标 A: Safety Margin
df["safety_margin"] = 1.0 - df["max_drift_m"] / INDUSTRY_REF["fence_radius_m"]

# 派生指标 B: 响应时间等级
def _rtl_grade(lat_ms):
    if lat_ms <= 10:    return "A (<10ms)"
    if lat_ms <= 100:   return "B (<100ms)"
    if lat_ms <= 1000:  return "C (<1s)"
    if lat_ms <= 3000:  return "D (<3s)"
    return "F (>3s)"
df["rtl_grade"] = df["latency_ms"].apply(_rtl_grade)


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


def _add_ref_line(ax, val, label, axis="h", color=PALETTE["red_strong"]):
    """添加行业参考线 (虚线 + 标签)"""
    if axis == "h":
        ax.axhline(val, color=color, ls="--", lw=0.9, alpha=0.7, zorder=1)
        ax.text(ax.get_xlim()[1], val, f" {label}", fontsize=8,
                color=color, va="bottom", ha="right", alpha=0.8)
    else:
        ax.axvline(val, color=color, ls="--", lw=0.9, alpha=0.7, zorder=1)
        ax.text(val, ax.get_ylim()[1] * 0.95, f" {label}", fontsize=8,
                color=color, va="top", ha="left", alpha=0.8, rotation=90)


# ═══════════════════════════════════════════════════════════════
# Fig 1 (a): 安全通过率对比 — Hero Panel
# ═══════════════════════════════════════════════════════════════
def plot_safety_pass_rate():
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    routes = ["hard_rule", "llm"]
    rates = [df[df["route"] == r]["safety_pass"].mean() * 100 for r in routes]
    counts = [df[df["route"] == r]["safety_pass"].sum() for r in routes]
    totals = [len(df[df["route"] == r]) for r in routes]
    colors = [ROUTE_PAL[r] for r in routes]
    labels = [ROUTE_LABELS[r] for r in routes]

    bars = ax.bar(range(len(routes)), rates, color=colors, width=0.5,
                  edgecolor="black", linewidth=0.8, alpha=0.9)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                f"{h:.0f}%\n({counts[i]}/{totals[i]})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(routes)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("安全通过率 (%)", fontsize=10)
    ax.set_title("急停安全通过率对比", fontsize=12, fontweight="bold", pad=6)
    ax.set_ylim(0, 115)



    add_panel_label(ax, "a")
    save(fig, "fig1_safety_pass_rate.png")


# ═══════════════════════════════════════════════════════════════
# Fig 2 (b): 响应延迟箱线图 (log scale + 行业参考线)
# ═══════════════════════════════════════════════════════════════
def plot_latency_comparison():
    fig, ax = plt.subplots(figsize=(COL_W, 2.8))

    data_hr = df[df["route"] == "hard_rule"]["latency_ms"].values
    data_llm = df[df["route"] == "llm"]["latency_ms"].values

    colors = [ROUTE_PAL["hard_rule"], ROUTE_PAL["llm"]]
    labels = [ROUTE_LABELS["hard_rule"], ROUTE_LABELS["llm"]]
    medians = [np.median(data_hr), np.median(data_llm)]
    positions = [1, 2]

    # Log-scale 柱状图（中位数）
    bars = ax.bar(positions, medians, width=0.5, color=colors,
                  alpha=0.55, edgecolor="black", linewidth=0.8, zorder=2)

    # 个体数据点叠加（Nature "show the data" 惯例）
    rng = np.random.default_rng(42)
    for data, pos, color in zip([data_hr, data_llm], positions, colors):
        jitter = rng.uniform(-0.1, 0.1, len(data))
        ax.scatter(pos + jitter, data, s=10, alpha=0.7,
                   color=color, edgecolor="white", linewidth=0.3, zorder=3)

    ax.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 5)  # 顶部留空给标注

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("响应延迟 (ms, log)", fontsize=10)
    ax.set_title("急停响应延迟对比", fontsize=12, fontweight="bold", pad=6)

    # 行业参考线
    _add_ref_line(ax, INDUSTRY_REF["px4_kill_ms"], "PX4 Kill <10ms", color=PALETTE["teal"])
    # SORA 参考线：标签放在线的上方，避免与散点重叠
    ax.axhline(INDUSTRY_REF["sora_window_ms"], color=PALETTE["red_strong"], ls="--", lw=0.9, alpha=0.7, zorder=1)
    ax.text(ax.get_xlim()[0] + 0.05, INDUSTRY_REF["sora_window_ms"] * 1.35,
            "SORA <3s", fontsize=8, color=PALETTE["red_strong"],
            va="bottom", ha="left", alpha=0.8)

    # 数值标注（柱顶上方）
    for i, (med, pos) in enumerate(zip(medians, positions)):
        ax.text(pos, med * 1.4, f"{med:.1f}ms", fontsize=9,
                ha="center", va="bottom", fontweight="bold", color=colors[i])

    add_panel_label(ax, "b")
    save(fig, "fig2_latency_comparison.png")


# ═══════════════════════════════════════════════════════════════
# Fig 3 (c): 最大漂移按扰动类型分组箱线图
# ═══════════════════════════════════════════════════════════════
def plot_drift_by_disturbance():
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))

    existing = [d for d in DISTURB_ORDER if d in df["disturbance"].unique()]
    n = len(existing)
    width = 0.35
    positions_hr = np.arange(n) * 2
    positions_llm = positions_hr + width + 0.05

    for route, positions, color in [
        ("hard_rule", positions_hr, ROUTE_PAL["hard_rule"]),
        ("llm", positions_llm, ROUTE_PAL["llm"]),
    ]:
        data = [df[(df["route"] == route) & (df["disturbance"] == d)]["max_drift_m"].values
                for d in existing]
        bp = ax.boxplot(
            data, positions=positions, widths=width,
            patch_artist=True, showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.4),
            medianprops=dict(color="black", linewidth=1),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)

    # 0.5m 安全阈值线
    _add_ref_line(ax, INDUSTRY_REF["drift_safe_m"], "安全阈值 0.5m", color=PALETTE["red_strong"])

    tick_positions = positions_hr + (width + 0.05) / 2
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([DISTURB_LABELS.get(d, d) for d in existing], fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("最大漂移 (m)", fontsize=10)
    ax.set_title("各扰动类型下急停漂移量", fontsize=12, fontweight="bold", pad=6)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ROUTE_PAL["hard_rule"], edgecolor="black", label=ROUTE_LABELS["hard_rule"]),
        Patch(facecolor=ROUTE_PAL["llm"], edgecolor="black", label=ROUTE_LABELS["llm"]),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    add_panel_label(ax, "c")
    save(fig, "fig3_drift_by_disturbance.png")


# ═══════════════════════════════════════════════════════════════
# Fig 4 (d): 安全通过率热力图 (disturbance × route)
# ═══════════════════════════════════════════════════════════════
def plot_safety_heatmap():
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    existing = [d for d in DISTURB_ORDER if d in df["disturbance"].unique()]
    routes = ["hard_rule", "llm"]

    matrix = np.zeros((len(routes), len(existing)))
    for i, r in enumerate(routes):
        for j, d in enumerate(existing):
            sub = df[(df["route"] == r) & (df["disturbance"] == d)]
            matrix[i, j] = sub["safety_pass"].mean() * 100 if len(sub) else 0

    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)

    # Annotations
    for i in range(len(routes)):
        for j in range(len(existing)):
            val = matrix[i, j]
            color = "white" if val > 65 else PALETTE["neutral_dark"]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color=color)

    ax.set_xticks(range(len(existing)))
    ax.set_xticklabels([DISTURB_LABELS.get(d, d) for d in existing], fontsize=6, rotation=30, ha="right")
    ax.set_yticks(range(len(routes)))
    ax.set_yticklabels([ROUTE_LABELS[r] for r in routes], fontsize=10)
    ax.set_title("各扰动类型安全通过率", fontsize=12, fontweight="bold", pad=6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("通过率 (%)", fontsize=9)

    add_panel_label(ax, "d")
    save(fig, "fig4_safety_heatmap.png")


# ═══════════════════════════════════════════════════════════════
# Fig 5 (e): 漂移量 CDF 对比 + 行业参考线
# ═══════════════════════════════════════════════════════════════
def plot_drift_cdf():
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    for route, color in ROUTE_PAL.items():
        data = np.sort(df[df["route"] == route]["max_drift_m"].values)
        cdf = np.arange(1, len(data) + 1) / len(data)
        ax.step(data, cdf, where="post", color=color, linewidth=1.5,
                label=ROUTE_LABELS[route], alpha=0.9)

    # 0.5m 垂直参考线
    ax.axvline(INDUSTRY_REF["drift_safe_m"], color=PALETTE["red_strong"],
               ls="--", lw=0.9, alpha=0.7)
    ax.text(INDUSTRY_REF["drift_safe_m"] + 0.05, 0.15, "安全阈值\n0.5m",
            fontsize=8, color=PALETTE["red_strong"], va="bottom")

    ax.set_xlabel("最大漂移 (m)", fontsize=10)
    ax.set_ylabel("累积概率", fontsize=10)
    ax.set_title("急停漂移量累积分布", fontsize=12, fontweight="bold", pad=6)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, max(df["max_drift_m"].max() * 1.1, 1))
    ax.set_ylim(0, 1.05)

    add_panel_label(ax, "e")
    save(fig, "fig5_drift_cdf.png")


# ═══════════════════════════════════════════════════════════════
# Fig 6 (f): 综合评估汇总表 (含派生指标)
# ═══════════════════════════════════════════════════════════════
def plot_summary_table():
    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    ax.axis("off")

    hr = df[df["route"] == "hard_rule"]
    llm = df[df["route"] == "llm"]

    # 响应时间等级
    hr_rtl = _rtl_grade(hr["latency_ms"].median())
    llm_rtl = _rtl_grade(llm["latency_ms"].median())

    # Safety Margin
    hr_sm = hr["safety_margin"].mean()
    llm_sm = llm["safety_margin"].mean()

    # 扰动敏感度 (DTS) — 各扰动下 drift 的标准差
    hr_dts = hr.groupby("disturbance")["max_drift_m"].std().mean()
    llm_dts = llm.groupby("disturbance")["max_drift_m"].std().mean()

    rows = [
        ["安全通过率 (%)", f"{hr['safety_pass'].mean()*100:.0f}", f"{llm['safety_pass'].mean()*100:.0f}"],
        ["中位延迟 (ms)", f"{hr['latency_ms'].median():.1f}", f"{llm['latency_ms'].median():.0f}"],
        ["响应等级 (RTL)", hr_rtl, llm_rtl],
        ["漂移均值 (m)", f"{hr['max_drift_m'].mean():.3f}", f"{llm['max_drift_m'].mean():.2f}"],
        ["漂移 P90 (m)", f"{hr['max_drift_m'].quantile(0.9):.3f}", f"{llm['max_drift_m'].quantile(0.9):.2f}"],
        ["最大速度均值 (m/s)", f"{hr['max_speed'].mean():.3f}", f"{llm['max_speed'].mean():.3f}"],
        ["安全裕度 SM", f"{hr_sm:.3f}", f"{llm_sm:.3f}"],
        ["扰动敏感度 DTS", f"{hr_dts:.4f}", f"{llm_dts:.4f}"],
        ["样本量 n", f"{len(hr)}", f"{len(llm)}"],
    ]

    col_labels = ["指标", "Ours (硬规则)", "Baseline (纯LLM)"]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Header styling
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(PALETTE["blue_main"])
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating rows
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#F0F4F8")
            else:
                cell.set_facecolor("white")

    ax.set_title("综合评估摘要（含行业对齐指标）", fontsize=12,
                 fontweight="bold", pad=15, y=0.95)
    add_panel_label(ax, "f", x=-0.05, y=1.02)
    save(fig, "fig6_summary_table.png")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("[Chart] Experiment 1 — Nature-Style Visualization")
    print(f"   Data:   {DATA_DIR}")
    print(f"   Output: {OUT_DIR}")
    print(f"   N={len(df)} | Routes: {df['route'].unique().tolist()}")
    print(f"   Width:  {COL_W}in (single-column)")
    print("=" * 55)

    plot_safety_pass_rate()
    plot_latency_comparison()
    plot_drift_by_disturbance()
    plot_safety_heatmap()
    plot_drift_cdf()
    plot_summary_table()

    print()
    print(f"Done! 6 figures × 3 formats (SVG + PNG + PDF) saved to: {OUT_DIR}")

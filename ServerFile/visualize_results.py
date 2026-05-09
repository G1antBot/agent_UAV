# -*- coding: utf-8 -*-
"""
Experiment 2 — Nature-Style Academic Visualization
Generates 8 publication-ready figures: SVG (primary) + PNG (300 DPI) + PDF (vector).
Single-column width: 89 mm ≈ 3.5 in.
"""

import os, json, sys, math
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

# ──────────────── Global Style (Nature Publication) ────────────────
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",        # editable text in SVG
    "pdf.fonttype": 42,            # TrueType in PDF
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi": 150,
    "font.size": 11,
    "legend.frameon": False,
    "legend.fontsize": 9,
})

# ── Nature PALETTE (semantic) ──
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "red_strong":     "#B64342",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "neutral_light":  "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
    "gold":           "#FFD700",
}
DEFAULT_COLORS = [
    PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"],
    PALETTE["teal"], PALETTE["violet"], PALETTE["neutral_light"],
]
DIFF_PAL = {"L1": PALETTE["blue_main"], "L2": PALETTE["teal"], "L3": PALETTE["red_strong"]}
SFS_PAL = [PALETTE["blue_main"], PALETTE["blue_secondary"], PALETTE["teal"], PALETTE["gold"]]

# Single-column width constant
COL_W = 3.5  # inches (89 mm)

# ──────────────── Path Resolution ────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EXP2_DIR = os.path.join(BASE_DIR, "logs", "test_exp2")

def _resolve_data_dir():
    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if not os.path.isabs(arg_path):
            arg_path = os.path.join(BASE_DIR, arg_path)
        if os.path.isdir(arg_path):
            return arg_path
        print(f"[WARN] Path not found: {arg_path}, auto-detecting...")
    if os.path.isdir(_DEFAULT_EXP2_DIR):
        run_dirs = sorted(
            [d for d in os.listdir(_DEFAULT_EXP2_DIR) if d.startswith("run_")],
            reverse=True,
        )
        if run_dirs:
            return os.path.join(_DEFAULT_EXP2_DIR, run_dirs[0])
    return _DEFAULT_EXP2_DIR

DATA_DIR = _resolve_data_dir()
CSV_PATH = os.path.join(DATA_DIR, "evaluation_results.csv")
JSON_PATH = os.path.join(DATA_DIR, "evaluation_metrics.json")
OUT_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────── Load Data (Pool all runs) ────────────────
def _load_pooled_data():
    """合并所有 run_* 目录下的 CSV，增加样本量以获得更稳健的统计量。"""
    import glob
    all_csvs = sorted(glob.glob(os.path.join(_DEFAULT_EXP2_DIR, "run_*", "evaluation_results.csv")))
    if len(all_csvs) > 1:
        frames = []
        for csv_path in all_csvs:
            run_name = os.path.basename(os.path.dirname(csv_path))
            tmp = pd.read_csv(csv_path)
            tmp["run_id"] = run_name
            frames.append(tmp)
        pooled = pd.concat(frames, ignore_index=True)
        print(f"  [POOL] Merged {len(all_csvs)} runs → {len(pooled)} total rows")
        return pooled
    # Fallback: single CSV
    return pd.read_csv(CSV_PATH)

df = _load_pooled_data()
try:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
except FileNotFoundError:
    metrics = {}

df["execution_success"] = df["execution_success"].astype(bool)
df["parsing_success"] = df["parsing_success"].astype(bool)
df["task_completed"] = df["task_completed"].astype(bool)
df["inference_latency_s"] = df["inference_latency_ms"] / 1000.0
df["execution_time_s"] = df["execution_time_ms"] / 1000.0

HAS_SFS_SUBS = all(c in df.columns for c in ["sfs_target", "sfs_param", "sfs_action", "sfs_execution"])

DIFF_ORDER = ["L1", "L2", "L3"]
TYPE_ORDER = ["基础运动", "目标定位", "语义修饰符", "条件分支", "复合任务"]

# ──────────────── Helpers ────────────────

def save(fig, name):
    """Triple-format output: SVG (primary) + PNG (300 DPI) + PDF (vector)."""
    fig.tight_layout(pad=0.5)
    base = name.replace(".png", "")
    for ext in (".svg", ".png", ".pdf"):
        p = os.path.join(OUT_DIR, base + ext)
        kw = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == ".png":
            kw["dpi"] = 300
        fig.savefig(p, **kw)
    plt.close(fig)
    print(f"  [OK] {base} (.svg + .png + .pdf)")


def add_panel_label(ax, label, x=-0.08, y=1.04):
    """Place a Nature-style bold lowercase panel label."""
    pass  # 不再绘制左上角字母标签


def is_dark(hex_color, threshold=128):
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < threshold


# ═══════════════════════════════════════════════════════════════
# Fig 1: TCR Heatmap
# ═══════════════════════════════════════════════════════════════
def plot_heatmap():
    core = df[df["group"] == "core"].copy()
    val_col = "task_completed" if "task_completed" in core.columns else "execution_success"
    pivot = core.pivot_table(
        index="difficulty", columns="semantic_type",
        values=val_col, aggfunc="mean"
    )
    pivot = pivot.reindex(index=[d for d in DIFF_ORDER if d in pivot.index],
                          columns=[t for t in TYPE_ORDER if t in pivot.columns])
    pivot = pivot * 100

    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", which="both", length=0)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            # luminance-aware text color
            norm_val = val / 100.0
            r, g, b, _ = plt.get_cmap("YlGnBu")(norm_val)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if lum < 0.5 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label("TCR (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_frame_on(False)
    ax.set_xlabel("语义类型", fontsize=10)
    ax.set_ylabel("难度", fontsize=10)
    ax.set_title("任务完成率 (TCR)：难度 × 语义类型", fontsize=12, fontweight="bold", pad=6)
    add_panel_label(ax, "a")
    save(fig, "fig1_tcr_heatmap.png")


# ═══════════════════════════════════════════════════════════════
# Fig 2: Latency Box Plot
# ═══════════════════════════════════════════════════════════════
def plot_latency_boxplot():
    core = df[df["group"] == "core"].copy()
    core["difficulty"] = pd.Categorical(core["difficulty"], categories=DIFF_ORDER, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.2), sharey=False)

    for idx, (col, title, label) in enumerate([
        ("inference_latency_s", "推理延迟", "b1"),
        ("execution_time_s", "执行耗时", "b2"),
    ]):
        data = [core[core["difficulty"] == d][col].dropna().values for d in DIFF_ORDER]
        bp = axes[idx].boxplot(
            data, patch_artist=True, widths=0.5,
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(markersize=3),
        )
        for patch, d in zip(bp["boxes"], DIFF_ORDER):
            patch.set_facecolor(DIFF_PAL[d])
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        axes[idx].set_xticklabels(
            [f"{d}\n(n={len(data[i])})" for i, d in enumerate(DIFF_ORDER)], fontsize=9
        )
        axes[idx].set_ylabel("时间 (s)", fontsize=10)
        axes[idx].set_title(title, fontsize=11, fontweight="bold")
        add_panel_label(axes[idx], label)

    save(fig, "fig2_latency_boxplot.png")


# ═══════════════════════════════════════════════════════════════
# Fig 3: Radar Chart
# ═══════════════════════════════════════════════════════════════
def plot_radar():
    m = metrics["models"][0]["metrics"]

    labels = ["PSR\n(解析成功率)", "TCR\n(任务完成率)", "CA\n(条件准确率)",
              "SFS\n(语义保真度)", "GR\n(泛化鲁棒性)", "EE\n(执行效率)"]
    values = [
        float(m["parsing_success_rate"].strip("%")),
        float(m["task_completion_rate"].strip("%")),
        float(m["conditional_accuracy"].strip("%")),
        float(m["semantic_fidelity_score"].strip("%") if "%" in m["semantic_fidelity_score"] else m["semantic_fidelity_score"]),
        float(m["generalization_robustness"].strip("%")),
        float(m["execution_efficiency"].strip("%")),
    ]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(COL_W, COL_W), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location("N")

    # Remove default grid, draw custom
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Custom concentric rings
    for r in [20, 40, 60, 80, 100]:
        ring = np.full(len(angles_closed), r)
        ax.plot(angles_closed, ring, color="#D8D8D8", lw=0.5, zorder=1)

    # Custom spokes
    for a in angles:
        ax.plot([a, a], [0, 105], color="#D8D8D8", lw=0.5, zorder=1)

    # Outer boundary
    ax.plot(angles_closed, np.full_like(angles_closed, 100.0),
            color="black", lw=0.8, zorder=2)

    # Data polygon
    ax.plot(angles_closed, values_closed, "o-", linewidth=1.8,
            color=PALETTE["blue_main"], markersize=4, zorder=4)
    ax.fill(angles_closed, values_closed, alpha=0.12, color=PALETTE["blue_main"], zorder=3)

    ax.set_ylim(0, 115)
    ax.set_yticks([])
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)

    # Value annotations
    for angle, val in zip(angles, values):
        ax.text(angle, val + 9, f"{val:.1f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color=PALETTE["blue_main"])

    ax.set_title("综合评估指标雷达图 (DeepSeek-V3)",
                 fontsize=12, fontweight="bold", pad=18)
    add_panel_label(ax, "c", x=-0.05, y=1.08)
    save(fig, "fig3_radar_metrics.png")


# ═══════════════════════════════════════════════════════════════
# Fig 4: Time Breakdown (horizontal stacked bar)
# ═══════════════════════════════════════════════════════════════
def plot_time_breakdown():
    core = df[df["group"] == "core"].copy()
    # 合并多次 run 的重复指令：按 instruction_id 取均值
    run_counts = core.groupby("instruction_id").size().rename("n_runs")
    agg_cols = {"inference_latency_s": "mean", "execution_time_s": "mean",
                "instruction_text": "first", "difficulty": "first"}
    core = core.groupby("instruction_id", as_index=False).agg(agg_cols)
    core = core.merge(run_counts, left_on="instruction_id", right_index=True)
    # 每个难度等级只保留测试次数最多的 5 条（最具统计代表性）
    core = (core.sort_values(["difficulty", "n_runs", "execution_time_s"],
                             ascending=[True, False, False])
                .groupby("difficulty").head(5)
                .sort_values("execution_time_s", ascending=True)
                .reset_index(drop=True))

    h_per_bar = 0.28
    fig_h = max(2.5, len(core) * h_per_bar + 0.8)
    fig, ax = plt.subplots(figsize=(COL_W + 1.0, fig_h))
    y_pos = np.arange(len(core))

    ax.barh(y_pos, core["inference_latency_s"], height=0.6,
            color=PALETTE["blue_main"], alpha=0.85, label="推理延迟",
            edgecolor="black", linewidth=0.5)
    ax.barh(y_pos, core["execution_time_s"], left=core["inference_latency_s"],
            height=0.6, color=PALETTE["green_3"], alpha=0.85, label="执行耗时",
            edgecolor="black", linewidth=0.5)

    labels = [f"[{row.difficulty}] {row.instruction_text[:7]}{'…' if len(row.instruction_text)>7 else ''}"
              for _, row in core.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)

    for idx, (_, row) in enumerate(core.iterrows()):
        total = row.inference_latency_s + row.execution_time_s
        ax.text(total + 0.3, idx, f"{total:.1f}s", va="center", fontsize=8,
                color=PALETTE["neutral_dark"])

    ax.set_xlabel("时间 (s)", fontsize=10)
    ax.set_title("各难度等级典型指令耗时分解",
                 fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=9, loc="lower right")
    add_panel_label(ax, "d")
    save(fig, "fig4_time_breakdown.png")


# ═══════════════════════════════════════════════════════════════
# Fig 5: Multi-Metric by Difficulty (grouped bar + dynamic Y)
# ═══════════════════════════════════════════════════════════════
def plot_difficulty_metrics():
    core = df[df["group"] == "core"].copy()
    core["difficulty"] = pd.Categorical(core["difficulty"], categories=DIFF_ORDER, ordered=True)

    metric_cols = {
        "TCR": "task_completed",
        "SFS": "semantic_fidelity_score",
        "EE": "execution_efficiency",
    }

    means, ci_lo, ci_hi = {}, {}, {}
    for label, col in metric_cols.items():
        if col == "task_completed":
            grouped = core.groupby("difficulty")[col].agg(["mean", "std", "count"])
            grouped["mean"] = grouped["mean"] * 100
            grouped["std"] = grouped["std"] * 100
        else:
            grouped = core.groupby("difficulty")[col].agg(["mean", "std", "count"])
        grouped = grouped.reindex(DIFF_ORDER)
        m = grouped["mean"].values
        # 95% CI = 1.96 * std / sqrt(n), clipped to [0, 100]
        ci = 1.96 * grouped["std"].fillna(0).values / np.sqrt(grouped["count"].values.clip(1))
        means[label] = m
        ci_lo[label] = np.clip(m - ci, 0, 100)   # lower bound
        ci_hi[label] = np.clip(m + ci, 0, 100)   # upper bound

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    x = np.arange(len(DIFF_ORDER))
    width = 0.22
    colors = [PALETTE["blue_main"], PALETTE["green_3"], PALETTE["gold"]]
    error_kw = {"elinewidth": 1.2, "capthick": 1.2, "capsize": 3}

    all_vals = []
    for i, (label, vals) in enumerate(means.items()):
        # Asymmetric error bars: [lower_err, upper_err]
        err_lower = vals - ci_lo[label]
        err_upper = ci_hi[label] - vals
        bars = ax.bar(x + i * width - width, vals, width,
                      label=label, color=colors[i], alpha=0.85,
                      edgecolor="black", linewidth=0.8,
                      yerr=[err_lower, err_upper], error_kw=error_kw)
        all_vals.extend(vals)
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 3,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Dynamic Y-axis
    y_min = max(0, min(all_vals) - 15)
    y_max = min(115, max(all_vals) + 15)
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={len(core[core['difficulty']==d])})" for d in DIFF_ORDER], fontsize=9)
    ax.set_ylabel("得分 / 比率 (%)", fontsize=10)
    ax.set_title("各难度等级多指标对比", fontsize=12, fontweight="bold", pad=6)
    ax.legend(fontsize=9, loc="upper right")
    ax.axhline(y=100, color=PALETTE["neutral_light"], linestyle="--", alpha=0.5, linewidth=0.8)
    add_panel_label(ax, "e")
    save(fig, "fig5_difficulty_metrics.png")


# ═══════════════════════════════════════════════════════════════
# Fig 6: Summary Table
# ═══════════════════════════════════════════════════════════════
def plot_summary_table():
    m = metrics["models"][0]["metrics"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.axis("off")

    table_data = [
        ["评估指标", "缩写", "数值"],
        ["解析成功率", "PSR", m["parsing_success_rate"]],
        ["任务完成率", "TCR", m["task_completion_rate"]],
        ["平均推理延迟", "Latency", f"{float(m['avg_inference_latency_ms']):.0f} ms"],
        ["条件分支准确率", "CA", m["conditional_accuracy"]],
        ["语义保真度评分", "SFS", m["semantic_fidelity_score"]],
        ["泛化鲁棒性", "GR", m["generalization_robustness"]],
        ["执行效率", "EE", m["execution_efficiency"]],
    ]

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc="center", loc="center", colWidths=[0.44, 0.14, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#E0E0E0")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(PALETTE["blue_main"])
            cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        elif row % 2 == 0:
            cell.set_facecolor("#F0F4FA")
        else:
            cell.set_facecolor("white")

    ax.set_title("DeepSeek-V3 综合评估摘要", fontsize=12, fontweight="bold", pad=12)
    add_panel_label(ax, "f", x=-0.02, y=0.98)
    save(fig, "fig6_summary_table.png")


# ═══════════════════════════════════════════════════════════════
# Fig 7: SFS Sub-Score Decomposition
# ═══════════════════════════════════════════════════════════════
def plot_sfs_breakdown():
    if not HAS_SFS_SUBS:
        print("  [SKIP] fig7_sfs_breakdown — SFS sub-scores not in CSV")
        return

    core = df[df["group"] == "core"].copy()
    # 合并多次 run 的重复指令：按 instruction_id 取均值
    run_counts = core.groupby("instruction_id").size().rename("n_runs")
    agg_cols = {"semantic_fidelity_score": "mean",
                "sfs_target": "mean", "sfs_param": "mean",
                "sfs_action": "mean", "sfs_execution": "mean",
                "instruction_text": "first", "difficulty": "first"}
    core = core.groupby("instruction_id", as_index=False).agg(agg_cols)
    core = core.merge(run_counts, left_on="instruction_id", right_index=True)
    # 每个难度等级只保留测试次数最多的 5 条
    core = (core.sort_values(["difficulty", "n_runs", "semantic_fidelity_score"],
                             ascending=[True, False, False])
                .groupby("difficulty").head(5)
                .sort_values("semantic_fidelity_score", ascending=True)
                .reset_index(drop=True))

    h_per_bar = 0.28
    fig_h = max(2.5, len(core) * h_per_bar + 0.8)
    fig, ax = plt.subplots(figsize=(COL_W + 1.0, fig_h))
    y_pos = np.arange(len(core))

    dims = [
        ("sfs_target", "目标准确性", SFS_PAL[0]),
        ("sfs_param", "参数精确性", SFS_PAL[1]),
        ("sfs_action", "动作正确性", SFS_PAL[2]),
        ("sfs_execution", "执行有效性", SFS_PAL[3]),
    ]

    left = np.zeros(len(core))
    for col, label, color in dims:
        vals = core[col].values.astype(float)
        ax.barh(y_pos, vals, left=left, height=0.6,
                color=color, alpha=0.85, label=label,
                edgecolor="black", linewidth=0.4)
        left += vals

    labels = [f"[{row.difficulty}] {row.instruction_text[:7]}{'…' if len(row.instruction_text)>7 else ''}"
              for _, row in core.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)

    for idx, (_, row) in enumerate(core.iterrows()):
        ax.text(row.semantic_fidelity_score + 1, idx,
                f"{row.semantic_fidelity_score:.0f}", va="center", fontsize=8,
                fontweight="bold", color=PALETTE["neutral_dark"])

    ax.set_xlabel("SFS 评分 (0–100)", fontsize=10)
    ax.set_xlim(0, 115)
    ax.axvline(x=100, color=PALETTE["neutral_light"], linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_title("语义保真度评分分解 (SFS, 均值, n=5)", fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=4, columnspacing=0.8, handlelength=1.2)
    add_panel_label(ax, "g")
    save(fig, "fig7_sfs_breakdown.png")


# ═══════════════════════════════════════════════════════════════
# Fig 8: Error Type Distribution (donut)
# ═══════════════════════════════════════════════════════════════
def plot_error_distribution():
    err_col = df["error_type"].fillna("N/A")
    counts = err_col.value_counts()

    success_count = counts.get("N/A", 0)
    error_counts = counts.drop("N/A", errors="ignore")

    if error_counts.empty:
        labels_all = ["任务完成"]
        sizes_all = [success_count]
        colors_all = [PALETTE["green_3"]]
    else:
        labels_all = ["任务完成"] + list(error_counts.index)
        sizes_all = [success_count] + list(error_counts.values)
        # Distinct colors for each error type to avoid visual overlap
        error_colors = [PALETTE["red_strong"], PALETTE["teal"], PALETTE["violet"],
                        PALETTE["neutral_mid"], PALETTE["blue_secondary"]]
        colors_all = [PALETTE["green_3"]] + [error_colors[i % len(error_colors)] for i in range(len(error_counts))]

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.8)) # Slightly shorter to fit legend better if needed, or keep COL_W
    
    # Create custom autopct to only show percentages > 5% inside the pie
    def custom_autopct(pct):
        return f"{pct:.0f}%" if pct > 5 else ""

    wedges, texts, autotexts = ax.pie(
        sizes_all, labels=None, autopct=custom_autopct,
        colors=colors_all, startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2)
    )
    
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")

    # Add legend to the right
    total = sum(sizes_all)
    legend_labels = [f"{l} ({s/total*100:.0f}%)" for l, s in zip(labels_all, sizes_all)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(0.9, 0.5), 
              fontsize=7, frameon=False, alignment="left")

    ax.set_title("执行结果分布", fontsize=9, fontweight="bold", pad=10)
    ax.text(0, 0, f"n={len(df)}", ha="center", va="center",
            fontsize=9, fontweight="bold", color=PALETTE["neutral_dark"])
    add_panel_label(ax, "h", x=-0.05, y=1.02)
    save(fig, "fig8_error_distribution.png")


# ═══════════════════════════════════════════════════════════════
# Fig 9: SFS vs EE Scatter (Relationship — Nature Level 3)
# ═══════════════════════════════════════════════════════════════
def plot_sfs_vs_ee_scatter():
    """Scatter: semantic fidelity vs execution efficiency, colored by difficulty.
    Answers 'how do quality and speed co-vary?' — Nature multi-panel Level 3."""
    core = df[df["group"] == "core"].copy()

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85))

    for d in DIFF_ORDER:
        sub = core[core["difficulty"] == d]
        ax.scatter(sub["semantic_fidelity_score"], sub["execution_efficiency"],
                   s=50, color=DIFF_PAL[d], edgecolors="white", linewidth=0.6,
                   alpha=0.85, label=d, zorder=5)

    # Median reference lines (quadrant analysis)
    med_sfs = core["semantic_fidelity_score"].median()
    med_ee = core["execution_efficiency"].median()
    ax.axvline(med_sfs, lw=0.8, ls="--", color=PALETTE["neutral_mid"], alpha=0.5)
    ax.axhline(med_ee, lw=0.8, ls="--", color=PALETTE["neutral_mid"], alpha=0.5)

    # Quadrant labels
    ax.text(106, 106, "高质·高效", fontsize=5, ha="right", va="top",
            color=PALETTE["neutral_mid"], style="italic")
    ax.text(106, -3, "高质·低效", fontsize=5, ha="right", va="bottom",
            color=PALETTE["neutral_mid"], style="italic")

    ax.set_xlabel("语义保真度 SFS", fontsize=10)
    ax.set_ylabel("执行效率 EE (%)", fontsize=10)
    ax.set_title("SFS 与执行效率的关联分析", fontsize=12, fontweight="bold", pad=6)
    ax.legend(fontsize=9, loc="lower left", markerscale=0.8)

    # Tighten
    ax.set_xlim(max(0, core["semantic_fidelity_score"].min() - 8), 108)
    ax.set_ylim(max(0, core["execution_efficiency"].min() - 8), 108)
    add_panel_label(ax, "i")
    save(fig, "fig9_sfs_vs_ee_scatter.png")


# ═══════════════════════════════════════════════════════════════
# Fig 10: Grouped Bar by Semantic Type
# ═══════════════════════════════════════════════════════════════
def plot_semantic_type_metrics():
    """Grouped bar: TCR & SFS by semantic type (different dimension from Fig 5)."""
    core = df[df["group"] == "core"].copy()

    existing_types = [t for t in TYPE_ORDER if t in core["semantic_type"].unique()]
    if not existing_types:
        print("  [SKIP] fig10 — no semantic types found")
        return

    metric_cols = {"TCR": "task_completed", "SFS": "semantic_fidelity_score"}
    means, ci_lo, ci_hi = {}, {}, {}
    for label, col in metric_cols.items():
        if col == "task_completed":
            grouped = core.groupby("semantic_type")[col].agg(["mean", "std", "count"])
            grouped["mean"] = grouped["mean"] * 100
            grouped["std"] = grouped["std"] * 100
        else:
            grouped = core.groupby("semantic_type")[col].agg(["mean", "std", "count"])
        grouped = grouped.reindex(existing_types)
        m = grouped["mean"].values
        ci = 1.96 * grouped["std"].fillna(0).values / np.sqrt(grouped["count"].values.clip(1))
        means[label] = m
        ci_lo[label] = np.clip(m - ci, 0, 100)
        ci_hi[label] = np.clip(m + ci, 0, 100)

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    x = np.arange(len(existing_types))
    width = 0.32
    colors = [PALETTE["blue_main"], PALETTE["green_3"]]
    error_kw = {"elinewidth": 1.0, "capthick": 1.0, "capsize": 3}

    all_vals = []
    for i, (label, vals) in enumerate(means.items()):
        err_lower = vals - ci_lo[label]
        err_upper = ci_hi[label] - vals
        bars = ax.bar(x + i * width - width / 2, vals, width,
                      label=label, color=colors[i], alpha=0.85,
                      edgecolor="black", linewidth=0.8,
                      yerr=[err_lower, err_upper], error_kw=error_kw)
        all_vals.extend(vals)
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    y_min = max(0, min(all_vals) - 15)
    y_max = min(115, max(all_vals) + 15)
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x + width / 4)
    short_labels = [t[:4] for t in existing_types]
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel("得分 (%)", fontsize=10)
    ax.set_title("各语义类型指标对比", fontsize=12, fontweight="bold", pad=6)
    ax.legend(fontsize=9, loc="lower left")
    ax.axhline(y=100, color=PALETTE["neutral_light"], linestyle="--", alpha=0.5, linewidth=0.8)
    add_panel_label(ax, "j")
    save(fig, "fig10_semantic_type_metrics.png")


# ═══════════════════════════════════════════════════════════════
# Fig 11: Core vs Zero-Shot Comparison
# ═══════════════════════════════════════════════════════════════
def plot_core_vs_zeroshot():
    """Alpha-graduated bar: core vs zero-shot group performance comparison.
    Validates the generalization claim (GR metric)."""
    zs = df[df["group"] == "zero_shot"]
    if zs.empty:
        print("  [SKIP] fig11 — no zero_shot data")
        return

    core = df[df["group"] == "core"]
    metrics_list = [
        ("TCR", "task_completed", True),
        ("SFS", "semantic_fidelity_score", False),
        ("EE", "execution_efficiency", False),
    ]

    core_vals, zs_vals = [], []
    labels = []
    for label, col, is_bool in metrics_list:
        if is_bool:
            core_vals.append(core[col].mean() * 100)
            zs_vals.append(zs[col].mean() * 100)
        else:
            core_vals.append(core[col].mean())
            zs_vals.append(zs[col].mean())
        labels.append(label)

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    x = np.arange(len(labels))
    width = 0.32

    # Core = full alpha, Zero-shot = lighter alpha (Pattern 5: ablation alpha encoding)
    blue_rgb = tuple(int(PALETTE["blue_main"].lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
    c_core = (*blue_rgb, 1.0)
    c_zs = (*blue_rgb, 0.45)

    bars1 = ax.bar(x - width / 2, core_vals, width, label=f"Core (n={len(core)})",
                   color=c_core, edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, zs_vals, width, label=f"Zero-shot (n={len(zs)})",
                   color=c_zs, edgecolor="black", linewidth=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    all_vals = core_vals + zs_vals
    y_min = max(0, min(all_vals) - 15)
    y_max = min(120, max(all_vals) + 15)
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("得分 (%)", fontsize=10)
    ax.set_title("Core 指令集 vs Zero-shot 泛化对比", fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=9, loc="lower right")
    ax.axhline(y=100, color=PALETTE["neutral_light"], linestyle="--", alpha=0.5, linewidth=0.8)
    add_panel_label(ax, "k")
    save(fig, "fig11_core_vs_zeroshot.png")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("[Chart] Experiment 2 — Nature-Style Visualization")
    print(f"   Data:   {DATA_DIR}")
    print(f"   Output: {OUT_DIR}")
    print(f"   N={len(df)} | SFS subs: {HAS_SFS_SUBS}")
    print(f"   Width:  {COL_W}in (single-column)")
    print("=" * 55)

    plot_heatmap()
    plot_latency_boxplot()
    plot_radar()
    plot_time_breakdown()
    plot_difficulty_metrics()
    plot_summary_table()
    plot_sfs_breakdown()
    plot_error_distribution()
    plot_sfs_vs_ee_scatter()
    plot_semantic_type_metrics()
    plot_core_vs_zeroshot()

    n_figs = 11 if HAS_SFS_SUBS else 10
    print()
    print(f"Done! {n_figs} figures × 3 formats (SVG + PNG + PDF) saved to: {OUT_DIR}")

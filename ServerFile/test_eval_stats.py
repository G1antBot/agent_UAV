import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime


def _load_rows(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["success"] = int(row.get("success", 0))
            except Exception:
                row["success"] = 0
            try:
                row["duration_s"] = float(row.get("duration_s", 0.0))
            except Exception:
                row["duration_s"] = 0.0
            rows.append(row)
    return rows


def _pick_run_id(rows, run_id=None):
    run_ids = [r.get("run_id", "") for r in rows if r.get("run_id")]
    if not run_ids:
        raise ValueError("No run_id found in CSV")
    if run_id:
        if run_id not in run_ids:
            raise ValueError(f"run_id {run_id} not found in CSV")
        return run_id
    # By default, use latest appended run_id
    return run_ids[-1]


def _calc_summary(rows, run_id):
    run_rows = [r for r in rows if r.get("run_id") == run_id]
    if not run_rows:
        raise ValueError(f"No rows for run_id={run_id}")

    module_stats = defaultdict(lambda: {"ok": 0, "total": 0, "dur_sum": 0.0})
    target_module_stats = defaultdict(lambda: defaultdict(lambda: {"ok": 0, "total": 0, "dur_sum": 0.0}))

    # Round-chain success: all tested modules in same (round, target) must succeed
    chain_group = defaultdict(list)

    for r in run_rows:
        module = r.get("module", "")
        target = r.get("target", "")
        round_id = r.get("round", "")
        success = int(r.get("success", 0))
        dur = float(r.get("duration_s", 0.0))

        module_stats[module]["total"] += 1
        module_stats[module]["ok"] += success
        module_stats[module]["dur_sum"] += dur

        target_module_stats[target][module]["total"] += 1
        target_module_stats[target][module]["ok"] += success
        target_module_stats[target][module]["dur_sum"] += dur

        chain_group[(round_id, target)].append(success)

    chain_total = len(chain_group)
    chain_ok = sum(1 for k in chain_group if all(v == 1 for v in chain_group[k]))

    return {
        "run_id": run_id,
        "row_count": len(run_rows),
        "chain_ok": chain_ok,
        "chain_total": chain_total,
        "module_stats": module_stats,
        "target_module_stats": target_module_stats,
    }


def _fmt_rate(ok, total):
    if total <= 0:
        return "0.0%"
    return f"{(ok * 100.0 / total):.1f}%"


def _build_markdown(summary):
    lines = []
    lines.append(f"# 评测统计报告（run_id={summary['run_id']}）")
    lines.append("")
    lines.append("## 1. 轮次链路成功率")
    lines.append("")
    lines.append(f"- 成功轮次: {summary['chain_ok']}")
    lines.append(f"- 总轮次: {summary['chain_total']}")
    lines.append(f"- 链路成功率: {_fmt_rate(summary['chain_ok'], summary['chain_total'])}")
    lines.append("")

    lines.append("## 2. 模块成功率总览")
    lines.append("")
    lines.append("| 模块 | 成功/总数 | 成功率 | 平均耗时(s) |")
    lines.append("|---|---:|---:|---:|")
    for module, s in summary["module_stats"].items():
        avg_dur = (s["dur_sum"] / s["total"]) if s["total"] else 0.0
        lines.append(
            f"| {module} | {s['ok']}/{s['total']} | {_fmt_rate(s['ok'], s['total'])} | {avg_dur:.2f} |"
        )
    lines.append("")

    lines.append("## 3. 分目标模块成功率")
    lines.append("")
    for target, module_map in summary["target_module_stats"].items():
        lines.append(f"### 目标: {target}")
        lines.append("")
        lines.append("| 模块 | 成功/总数 | 成功率 | 平均耗时(s) |")
        lines.append("|---|---:|---:|---:|")
        for module, s in module_map.items():
            avg_dur = (s["dur_sum"] / s["total"]) if s["total"] else 0.0
            lines.append(
                f"| {module} | {s['ok']}/{s['total']} | {_fmt_rate(s['ok'], s['total'])} | {avg_dur:.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize eval_summary.csv for PPT-ready metrics.")
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "logs", "test_eval", "eval_summary.csv"),
        help="Path to eval_summary.csv",
    )
    parser.add_argument("--run-id", default="", help="Specific run_id to summarize. Default: latest run_id in csv.")
    args = parser.parse_args()

    rows = _load_rows(args.csv)
    run_id = _pick_run_id(rows, args.run_id or None)
    summary = _calc_summary(rows, run_id)

    out_dir = os.path.dirname(args.csv)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"eval_stats_{ts}_{run_id}.json")
    md_path = os.path.join(out_dir, f"eval_stats_{ts}_{run_id}.md")

    # Convert defaultdicts to normal dicts for JSON dump
    serializable = {
        "run_id": summary["run_id"],
        "row_count": summary["row_count"],
        "chain_ok": summary["chain_ok"],
        "chain_total": summary["chain_total"],
        "module_stats": {k: dict(v) for k, v in summary["module_stats"].items()},
        "target_module_stats": {
            target: {module: dict(stats) for module, stats in module_map.items()}
            for target, module_map in summary["target_module_stats"].items()
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    md_text = _build_markdown(summary)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print("=== 统计完成 ===")
    print(f"run_id: {run_id}")
    print(f"json: {json_path}")
    print(f"md  : {md_path}")


if __name__ == "__main__":
    main()

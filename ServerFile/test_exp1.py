import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
import random

import numpy as np

sys.path.append(r"D:\Rflysim\RflySimAPIs\RflySimSDK\vision")

from Communication_Mavlink import BodyCommMavlink
from OpenAI_api_Mavlink_Agent import OpenAI_APIs
from runtime_logger import get_run_id, get_runtime_logger, init_runtime_logger
from smolagents import (
    CodeAgent,
    FinalAnswerPromptTemplate,
    ManagedAgentPromptTemplate,
    PlanningPromptTemplate,
    PromptTemplates,
)
from volcEngineLLM import VolcEngineFakeHFModel


def _safe_stop(comm_api):
    try:
        comm_api.MavList[0].SendVelFRD(0, 0, 0, 0)
    except Exception:
        pass


def _auto_offboard_takeoff(comm_api, logger, takeoff_height=-0.5, arm_wait=5.0, hold_wait=5.0):
    if not comm_api.MavList:
        raise RuntimeError("No MAV controller found for auto takeoff")

    mav = comm_api.MavList[0]
    logger.info(
        f"AUTO_TAKEOFF_START target_height={takeoff_height} arm_wait={arm_wait} hold_wait={hold_wait}"
    )

    mav.initOffboard()
    time.sleep(max(arm_wait, 0.0))

    cur_x = float(mav.uavPosNED[0])
    cur_y = float(mav.uavPosNED[1])
    cur_yaw = float(mav.uavAngEular[2])
    mav.SendPosNED(cur_x, cur_y, float(takeoff_height), cur_yaw)
    time.sleep(max(hold_wait, 0.0))

    _safe_stop(comm_api)
    logger.info("AUTO_TAKEOFF_DONE")


def _return_to_home_pose(comm_api, logger, home_pose, settle_wait=3.0, approach_gap=2.0):
    if not comm_api.MavList:
        raise RuntimeError("No MAV controller found for return-home")
    if not home_pose:
        raise RuntimeError("home_pose is empty")

    mav = comm_api.MavList[0]
    tx, ty, tz, tyaw = home_pose
    current_x = float(mav.uavPosNED[0])
    current_y = float(mav.uavPosNED[1])
    current_z = float(mav.uavPosNED[2])
    dx = tx - current_x
    dy = ty - current_y
    distance = (dx * dx + dy * dy) ** 0.5
    if distance <= max(approach_gap, 0.0):
        mid_x, mid_y = tx, ty
    else:
        ratio = max(distance - approach_gap, 0.0) / distance
        mid_x = current_x + dx * ratio
        mid_y = current_y + dy * ratio

    logger.info(
        f"RETURN_HOME_START x={tx:.3f} y={ty:.3f} z={tz:.3f} yaw={tyaw:.3f} mid_x={mid_x:.3f} mid_y={mid_y:.3f}"
    )
    mav.SendPosNED(float(mid_x), float(mid_y), float(current_z), float(tyaw))
    time.sleep(max(settle_wait, 0.0))
    mav.SendPosNED(float(tx), float(ty), float(tz), float(tyaw))
    time.sleep(max(settle_wait, 0.0))
    _safe_stop(comm_api)
    logger.info("RETURN_HOME_DONE")


def _build_agent(chat_api):
    prompt_templates = PromptTemplates(
        system_prompt=chat_api.Prompt_dit["Prompt_smol"],
        planning=PlanningPromptTemplate(
            initial_plan="",
            update_plan_pre_messages="",
            update_plan_post_messages="",
        ),
        managed_agent=ManagedAgentPromptTemplate(
            task="",
            report="",
        ),
        final_answer=FinalAnswerPromptTemplate(
            pre_messages="",
            post_messages="",
        ),
    )
    agent = CodeAgent(model="deepseek-v3", prompt_templates=prompt_templates, tools=[])
    agent.model = VolcEngineFakeHFModel()
    return agent


def _send_motion_for_stress(comm_api, speed=0.8, duration_s=1.2, step_s=0.05):
    mav = comm_api.MavList[0]
    t0 = time.monotonic()
    while (time.monotonic() - t0) < max(duration_s, 0.0):
        mav.SendVelFRD(float(speed), 0.0, 0.0, 0.0)
        time.sleep(max(step_s, 0.01))


def _send_random_disturbance(comm_api, speed=0.8, yaw_rate=0.5, duration_s=1.0, step_s=0.05):
    mav = comm_api.MavList[0]
    mode = random.choice(["forward", "left", "right", "back", "yaw_left", "yaw_right", "hover"])

    if mode == "forward":
        vx, vy, vz, yawrate = float(speed), 0.0, 0.0, 0.0
    elif mode == "back":
        vx, vy, vz, yawrate = -float(speed), 0.0, 0.0, 0.0
    elif mode == "left":
        vx, vy, vz, yawrate = 0.0, -float(speed), 0.0, 0.0
    elif mode == "right":
        vx, vy, vz, yawrate = 0.0, float(speed), 0.0, 0.0
    elif mode == "yaw_left":
        vx, vy, vz, yawrate = 0.0, 0.0, 0.0, float(yaw_rate)
    elif mode == "yaw_right":
        vx, vy, vz, yawrate = 0.0, 0.0, 0.0, -float(yaw_rate)
    else:
        vx, vy, vz, yawrate = 0.0, 0.0, 0.0, 0.0

    t0 = time.monotonic()
    while (time.monotonic() - t0) < max(duration_s, 0.0):
        mav.SendVelFRD(vx, vy, vz, yawrate)
        time.sleep(max(step_s, 0.01))

    return mode


def _run_conflict_flow(chat_api, comm_api, use_hard_rule):
    commands = [
        {"text": "向前飞0.3米", "vx": 0.6, "vy": 0.0, "vz": 0.0, "yawrate": 0.0, "duration": 0.5},
        {"text": "向左飞0.3米", "vx": 0.0, "vy": -0.6, "vz": 0.0, "yawrate": 0.0, "duration": 0.5},
        {"text": "向右飞0.3米", "vx": 0.0, "vy": 0.6, "vz": 0.0, "yawrate": 0.0, "duration": 0.5},
    ]

    if use_hard_rule:
        for cmd in commands:
            chat_api._handle_hard_rules(cmd["text"])
        return "hard_rule"

    mav = comm_api.MavList[0]
    for cmd in commands:
        t0 = time.monotonic()
        while (time.monotonic() - t0) < max(cmd["duration"], 0.0):
            mav.SendVelFRD(cmd["vx"], cmd["vy"], cmd["vz"], cmd["yawrate"])
            time.sleep(0.05)
    return "direct"


def _sample_stability(comm_api, trigger_pos, window_s, sample_s, drift_threshold_m, speed_threshold):
    mav = comm_api.MavList[0]
    max_drift = 0.0
    max_speed = 0.0
    stable_start = None
    stable_need = 3
    stable_hits = 0

    start_ts = time.monotonic()
    while (time.monotonic() - start_ts) < max(window_s, 0.0):
        pos = np.array(mav.uavPosNED[:3], dtype=float)
        drift = float(np.linalg.norm(pos - trigger_pos))
        max_drift = max(max_drift, drift)

        vel = getattr(mav, "uavVelNED", None)
        if vel is not None:
            try:
                speed = float(np.linalg.norm(np.array(vel[:3], dtype=float)))
            except Exception:
                speed = 0.0
        else:
            speed = 0.0
        max_speed = max(max_speed, speed)

        if drift <= drift_threshold_m and speed <= speed_threshold:
            stable_hits += 1
            if stable_hits >= stable_need and stable_start is None:
                stable_start = time.monotonic()
        else:
            stable_hits = 0

        time.sleep(max(sample_s, 0.01))

    stabilize_s = None
    if stable_start is not None:
        stabilize_s = stable_start - start_ts

    return max_drift, max_speed, stabilize_s


def _trigger_emergency(chat_api, agent, stop_text, use_hard_rule):
    start = time.perf_counter()
    if use_hard_rule:
        action, summary = chat_api._handle_hard_rules(stop_text)
        ok = action == "continue"
        route = "hard_rule"
    else:
        ok = bool(chat_api._run_agent_for_clause(agent, stop_text))
        summary = chat_api._get_latest_result_cn(default_text="LLM路径执行完成" if ok else "LLM路径执行失败")
        route = "llm"
    latency_ms = (time.perf_counter() - start) * 1000.0
    return ok, summary, latency_ms, route


def _run_one_trial(
    trial_idx,
    chat_api,
    comm_api,
    logger,
    home_pose,
    stop_text,
    use_hard_rule,
    agent,
    settle_wait_s,
    motion_speed,
    motion_duration_s,
    drift_threshold_m,
    stable_window_s,
    stable_sample_s,
    speed_threshold,
    conflict_flow,
):
    _return_to_home_pose(comm_api, logger, home_pose, settle_wait=2.0, approach_gap=1.5)

    disturbance_mode = _send_random_disturbance(
        comm_api,
        speed=motion_speed,
        yaw_rate=max(motion_speed * 0.8, 0.2),
        duration_s=motion_duration_s,
    )
    trigger_pos = np.array(comm_api.MavList[0].uavPosNED[:3], dtype=float)

    conflict_route = ""
    if conflict_flow:
        conflict_route = _run_conflict_flow(chat_api, comm_api, use_hard_rule)

    ok, summary, latency_ms, route = _trigger_emergency(
        chat_api,
        agent,
        stop_text,
        use_hard_rule=use_hard_rule,
    )

    time.sleep(max(settle_wait_s, 0.0))
    max_drift, max_speed, stabilize_s = _sample_stability(
        comm_api,
        trigger_pos,
        window_s=stable_window_s,
        sample_s=stable_sample_s,
        drift_threshold_m=drift_threshold_m,
        speed_threshold=speed_threshold,
    )
    safety_pass = bool(ok) and (max_drift <= drift_threshold_m) and (max_speed <= speed_threshold)
    _safe_stop(comm_api)

    logger.info(
        f"EXP1_TRIAL idx={trial_idx} route={route} ok={ok} safety_pass={safety_pass} "
        f"latency_ms={latency_ms:.2f} max_drift_m={max_drift:.3f} max_speed={max_speed:.3f} "
        f"stabilize_s={stabilize_s} disturbance={disturbance_mode} conflict={conflict_route} summary={summary}"
    )

    return {
        "run_id": get_run_id(),
        "trial": int(trial_idx),
        "route": route,
        "disturbance": disturbance_mode,
        "conflict": conflict_route,
        "success": int(bool(ok)),
        "latency_ms": round(latency_ms, 3),
        "max_drift_m": round(max_drift, 4),
        "max_speed": round(max_speed, 4),
        "stabilize_s": round(float(stabilize_s), 4) if stabilize_s is not None else "",
        "safety_pass": int(bool(safety_pass)),
        "summary": summary,
    }


def _save_outputs(output_dir, run_id, rows, route_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"exp1_summary_{ts}_{run_id}.json")
    csv_path = os.path.join(output_dir, "exp1_trials.csv")
    md_path = os.path.join(output_dir, f"exp1_report_{ts}_{run_id}.md")

    latencies = [r["latency_ms"] for r in rows]
    drifts = [r["max_drift_m"] for r in rows]
    speeds = [r["max_speed"] for r in rows]
    stabilize_vals = [r["stabilize_s"] for r in rows if isinstance(r["stabilize_s"], (int, float))]
    success_count = sum(r["success"] for r in rows)
    safety_count = sum(r["safety_pass"] for r in rows)
    total = len(rows)
    failure_rate = (1.0 - (safety_count / total)) if total else 1.0

    summary = {
        "run_id": run_id,
        "route": route_name,
        "trials": total,
        "success_count": success_count,
        "success_rate": round((success_count / total) * 100.0, 2) if total else 0.0,
        "safety_pass_count": safety_count,
        "safety_pass_rate": round((safety_count / total) * 100.0, 2) if total else 0.0,
        "failure_rate": round(failure_rate * 100.0, 2),
        "latency_ms": {
            "avg": round(float(np.mean(latencies)), 3) if latencies else 0.0,
            "p50": round(float(np.percentile(latencies, 50)), 3) if latencies else 0.0,
            "p90": round(float(np.percentile(latencies, 90)), 3) if latencies else 0.0,
            "p95": round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
            "max": round(float(np.max(latencies)), 3) if latencies else 0.0,
        },
        "max_drift_m": {
            "avg": round(float(np.mean(drifts)), 3) if drifts else 0.0,
            "p90": round(float(np.percentile(drifts, 90)), 3) if drifts else 0.0,
            "max": round(float(np.max(drifts)), 3) if drifts else 0.0,
        },
        "max_speed": {
            "avg": round(float(np.mean(speeds)), 3) if speeds else 0.0,
            "p90": round(float(np.percentile(speeds, 90)), 3) if speeds else 0.0,
            "max": round(float(np.max(speeds)), 3) if speeds else 0.0,
        },
        "stabilize_s": {
            "avg": round(float(np.mean(stabilize_vals)), 3) if stabilize_vals else 0.0,
            "p90": round(float(np.percentile(stabilize_vals, 90)), 3) if stabilize_vals else 0.0,
            "max": round(float(np.max(stabilize_vals)), 3) if stabilize_vals else 0.0,
        },
        "rows": rows,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "trial",
                "route",
                "disturbance",
                "conflict",
                "success",
                "latency_ms",
                "max_drift_m",
                "max_speed",
                "stabilize_s",
                "safety_pass",
                "summary",
            ],
        )
        if not csv_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        f"# 实验一统计报告（run_id={run_id}）",
        "",
        f"- 路由模式: {route_name}",
        f"- 测试次数: {total}",
        f"- 急停成功率: {summary['success_rate']:.2f}% ({success_count}/{total})",
        f"- 安全通过率: {summary['safety_pass_rate']:.2f}% ({safety_count}/{total})",
        f"- 失败率: {summary['failure_rate']:.2f}%",
        "",
        "## 延迟统计(ms)",
        "",
        f"- 平均: {summary['latency_ms']['avg']}",
        f"- P50: {summary['latency_ms']['p50']}",
        f"- P90: {summary['latency_ms']['p90']}",
        f"- P95: {summary['latency_ms']['p95']}",
        f"- 最大: {summary['latency_ms']['max']}",
        "",
        "## 急停后稳定性统计",
        "",
        f"- 最大漂移均值(m): {summary['max_drift_m']['avg']}",
        f"- 最大漂移P90(m): {summary['max_drift_m']['p90']}",
        f"- 最大漂移峰值(m): {summary['max_drift_m']['max']}",
        f"- 最大速度均值(m/s): {summary['max_speed']['avg']}",
        f"- 最大速度P90(m/s): {summary['max_speed']['p90']}",
        f"- 最大速度峰值(m/s): {summary['max_speed']['max']}",
        f"- 稳定时间均值(s): {summary['stabilize_s']['avg']}",
        f"- 稳定时间P90(s): {summary['stabilize_s']['p90']}",
        f"- 稳定时间峰值(s): {summary['stabilize_s']['max']}",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return json_path, csv_path, md_path, summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: architecture ablation for emergency-stop.")
    parser.add_argument("--trials", type=int, default=50, help="Number of emergency-stop trials.")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "logs", "test_exp1"))
    parser.add_argument("--init-wait", type=float, default=5.0)
    parser.add_argument("--takeoff-height", type=float, default=-2.0)
    parser.add_argument("--arm-wait", type=float, default=5.0)
    parser.add_argument("--takeoff-wait", type=float, default=5.0)
    parser.add_argument("--settle-after-stop", type=float, default=0.3)
    parser.add_argument("--stress-speed", type=float, default=0.8)
    parser.add_argument("--stress-duration", type=float, default=1.2)
    parser.add_argument("--drift-threshold", type=float, default=0.50)
    parser.add_argument("--stable-window", type=float, default=2.0)
    parser.add_argument("--stable-sample", type=float, default=0.1)
    parser.add_argument("--speed-threshold", type=float, default=0.65)
    parser.add_argument("--stop-text", default="急停")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--conflict-flow", action="store_true")
    args = parser.parse_args()

    if args.trials <= 0:
        raise ValueError("trials must be > 0")

    os.makedirs(args.output_dir, exist_ok=True)
    init_runtime_logger(log_dir=args.output_dir)
    logger = get_runtime_logger("test_exp1")

    logger.info(f"EXP1_START trials={args.trials} output_dir={args.output_dir}")
    random.seed(args.seed)

    comm_api = BodyCommMavlink()
    time.sleep(max(args.init_wait, 0.0))

    preflight = comm_api.preflight_check()
    logger.info(f"EXP1_PREFLIGHT {preflight}")
    if not bool(preflight.get("ok", False)):
        raise RuntimeError("实验一预检失败，请先修复链路/状态")

    _auto_offboard_takeoff(
        comm_api,
        logger,
        takeoff_height=args.takeoff_height,
        arm_wait=args.arm_wait,
        hold_wait=args.takeoff_wait,
    )

    home_pose = (
        float(comm_api.MavList[0].uavPosNED[0]),
        float(comm_api.MavList[0].uavPosNED[1]),
        float(args.takeoff_height),
        float(comm_api.MavList[0].uavAngEular[2]),
    )

    chat_api = OpenAI_APIs(
        comm_api.MavList,
        len(comm_api.MavList),
        comm_api.detect_yolo,
        comm_api.approachObjective,
        comm_api.look,
        comm_api.search_object,
        comm_api.save_detection_image,
        comm_api.face_objective_to_target,
        comm_api.strike_objective_to_target,
    )

    use_hard_rule = bool(getattr(chat_api, "enable_hard_rule_routing", True))
    route_name = "ours_hard_rule" if use_hard_rule else "base_pure_llm"
    logger.info(f"EXP1_ROUTE route={route_name} enable_hard_rule_routing={use_hard_rule}")

    agent = None
    if not use_hard_rule:
        agent = _build_agent(chat_api)

    rows = []
    try:
        for i in range(1, args.trials + 1):
            row = _run_one_trial(
                trial_idx=i,
                chat_api=chat_api,
                comm_api=comm_api,
                logger=logger,
                home_pose=home_pose,
                stop_text=args.stop_text,
                use_hard_rule=use_hard_rule,
                agent=agent,
                settle_wait_s=args.settle_after_stop,
                motion_speed=args.stress_speed,
                motion_duration_s=args.stress_duration,
                drift_threshold_m=args.drift_threshold,
                stable_window_s=args.stable_window,
                stable_sample_s=args.stable_sample,
                speed_threshold=args.speed_threshold,
                conflict_flow=args.conflict_flow,
            )
            rows.append(row)
    finally:
        _safe_stop(comm_api)
        try:
            comm_api.close_image_source()
        except Exception:
            pass

    json_path, csv_path, md_path, summary = _save_outputs(args.output_dir, get_run_id(), rows, route_name)

    print("=== EXP1 DONE ===")
    print(f"route: {route_name}")
    print(f"trials: {summary['trials']}")
    print(f"success_rate: {summary['success_rate']}%")
    print(f"safety_pass_rate: {summary['safety_pass_rate']}%")
    print(f"failure_rate: {summary['failure_rate']}%")
    print(f"avg_latency_ms: {summary['latency_ms']['avg']}")
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")
    print(f"md  : {md_path}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

sys.path.append(r"D:\Rflysim\RflySimAPIs\RflySimSDK\vision")

from Communication_Mavlink import BodyCommMavlink
from runtime_logger import get_run_id, get_runtime_logger, init_runtime_logger


def _safe_stop(comm_api):
    """Try to stop the drone safely after each test item."""
    try:
        comm_api.MavList[0].SendVelFRD(0, 0, 0, 0)
    except Exception:
        pass


def _auto_offboard_takeoff(comm_api, logger, takeoff_height=-0.5, arm_wait=5.0, hold_wait=5.0):
    """Auto enter offboard mode and take off before evaluation."""
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
    """Return drone to the recorded home pose before each module test.

    Uses a two-segment path to reduce overshoot:
    1) move to a point `approach_gap` meters before home
    2) move from that point to the exact home pose
    """
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
        f"RETURN_HOME_START x={tx:.3f} y={ty:.3f} z={tz:.3f} yaw={tyaw:.3f} "
        f"mid_x={mid_x:.3f} mid_y={mid_y:.3f} settle_wait={settle_wait}"
    )

    # 先飞到 home 前方的中间点，避免一步到位冲过头
    mav.SendPosNED(float(mid_x), float(mid_y), float(current_z), float(tyaw))
    time.sleep(max(settle_wait, 0.0))

    # 再精确回到 home 点
    mav.SendPosNED(float(tx), float(ty), float(tz), float(tyaw))
    time.sleep(max(settle_wait, 0.0))

    _safe_stop(comm_api)
    logger.info("RETURN_HOME_DONE")


def _run_one_test(name, test_fn, logger, comm_api, round_idx, target):
    start = time.time()
    success = False
    error = ""

    logger.info(f"TEST_START round={round_idx} target={target} module={name}")
    try:
        success = bool(test_fn())
    except Exception as exc:
        success = False
        error = str(exc)
        logger.exception(f"TEST_EXCEPTION round={round_idx} target={target} module={name} err={error}")

    duration = time.time() - start
    summary = getattr(comm_api, "last_search_result_cn", "") or ""
    _safe_stop(comm_api)

    logger.info(
        f"TEST_END round={round_idx} target={target} module={name} success={success} duration_s={duration:.3f} summary={summary} err={error}"
    )
    return {
        "round": round_idx,
        "target": target,
        "module": name,
        "success": success,
        "duration_s": round(duration, 3),
        "summary": summary,
        "error": error,
    }


def _save_results(log_dir, run_id, search_mode, rounds, targets, results):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(log_dir, f"eval_summary_{ts}_{run_id}.json")
    csv_path = os.path.join(log_dir, "eval_summary.csv")

    payload = {
        "run_id": run_id,
        "rounds": rounds,
        "targets": targets,
        "search_mode": search_mode,
        "results": results,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "run_id",
                "round",
                "target",
                "search_mode",
                "module",
                "success",
                "duration_s",
                "summary",
                "error",
            ])
        for item in results:
            writer.writerow([
                run_id,
                item["round"],
                item["target"],
                search_mode,
                item["module"],
                int(item["success"]),
                item["duration_s"],
                item["summary"],
                item["error"],
            ])

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate search/approach/strike modules by rounds.")
    parser.add_argument(
        "--targets",
        default="car,red balloon,uav,blue ball",
        help="Comma-separated target list. Round n uses targets[(n-1) % len(targets)].",
    )
    parser.add_argument("--rounds", type=int, default=15, help="Total evaluation rounds.")
    parser.add_argument(
        "--search-mode",
        default="quick",
        choices=["quick", "all"],
        help="Search mode for search module.",
    )
    parser.add_argument(
        "--modules",
        default="search,approach,strike",
        help="Comma-separated module list from: search,approach,strike",
    )
    parser.add_argument("--init-wait", type=float, default=5.0, help="Wait seconds after comm init.")
    parser.add_argument("--takeoff-height", type=float, default=-1.0, help="Auto takeoff target height in NED.")
    parser.add_argument("--arm-wait", type=float, default=5.0, help="Wait seconds after initOffboard.")
    parser.add_argument("--takeoff-wait", type=float, default=5.0, help="Wait seconds after SendPosNED takeoff command.")
    parser.add_argument("--module-settle-wait", type=float, default=3.0, help="Wait seconds after returning home before each module.")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    eval_log_dir = os.path.join(base_dir, "logs", "test_eval")
    os.makedirs(eval_log_dir, exist_ok=True)

    init_runtime_logger(log_dir=eval_log_dir)
    logger = get_runtime_logger("test")
    target_list = [x.strip() for x in args.targets.split(",") if x.strip()]
    if not target_list:
        raise ValueError("No valid targets provided")
    if args.rounds <= 0:
        raise ValueError("rounds must be > 0")

    logger.info(
        f"评测启动 rounds={args.rounds} targets={target_list} search_mode={args.search_mode} modules={args.modules}"
    )

    comm_api = BodyCommMavlink()
    logger.info("通信模块初始化完成")
    time.sleep(max(args.init_wait, 0.0))
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
    logger.info(
        f"HOME_POSE_SET x={home_pose[0]:.3f} y={home_pose[1]:.3f} z={home_pose[2]:.3f} yaw={home_pose[3]:.3f}"
    )

    selected = [x.strip().lower() for x in args.modules.split(",") if x.strip()]
    valid = {"search", "approach", "strike"}
    selected = [x for x in selected if x in valid]
    if not selected:
        raise ValueError("No valid modules selected. Use search,approach,strike")

    results = []
    for round_idx in range(1, args.rounds + 1):
        target = target_list[(round_idx - 1) % len(target_list)]
        logger.info(f"ROUND_START round={round_idx}/{args.rounds} target={target}")

        if "search" in selected:
            _return_to_home_pose(comm_api, logger, home_pose, settle_wait=args.module_settle_wait)
            results.append(
                _run_one_test(
                    "search",
                    lambda t=target: comm_api.search_object(t, mode=args.search_mode),
                    logger,
                    comm_api,
                    round_idx,
                    target,
                )
            )

        if "approach" in selected:
            _return_to_home_pose(comm_api, logger, home_pose, settle_wait=args.module_settle_wait)
            results.append(
                _run_one_test(
                    "approach",
                    lambda t=target: comm_api.approach_objective_to_target(t),
                    logger,
                    comm_api,
                    round_idx,
                    target,
                )
            )

        if "strike" in selected:
            _return_to_home_pose(comm_api, logger, home_pose, settle_wait=args.module_settle_wait)
            results.append(
                _run_one_test(
                    "strike",
                    lambda t=target: comm_api.strike_objective_to_target(t),
                    logger,
                    comm_api,
                    round_idx,
                    target,
                )
            )

        logger.info(f"ROUND_END round={round_idx}/{args.rounds} target={target}")

    run_id = get_run_id()
    json_path, csv_path = _save_results(
        eval_log_dir,
        run_id,
        args.search_mode,
        args.rounds,
        target_list,
        results,
    )

    print("\n=== EVAL RESULT ===")
    for item in results:
        print(
            f"round={item['round']} target={item['target']} module={item['module']}: "
            f"success={item['success']} duration_s={item['duration_s']} summary={item['summary']}"
        )
    print(f"JSON summary: {json_path}")
    print(f"CSV summary : {csv_path}")


if __name__ == "__main__":
    main()

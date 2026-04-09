import argparse
import csv
import os
import sys
import time
from datetime import datetime

sys.path.append(r"D:\Rflysim\RflySimAPIs\RflySimSDK\vision")

from Communication_Mavlink import BodyCommMavlink
from OpenAI_api_Mavlink_Agent import OpenAI_APIs
from runtime_logger import get_run_id, get_runtime_logger, init_runtime_logger
from smolagents import CodeAgent, FinalAnswerPromptTemplate, ManagedAgentPromptTemplate, PlanningPromptTemplate, PromptTemplates
from volcEngineLLM import VolcEngineFakeHFModel


MODULE_BLUEPRINTS = [
    {"module": "search", "verb": "搜索", "search_mode": "quick"},
    {"module": "approach", "verb": "靠近", "search_mode": "quick"},
    {"module": "strike", "verb": "打击", "search_mode": "quick"},
]

TARGETS = [
    {"label": "red balloon", "prompt": "红色气球"},
    {"label": "blue ball", "prompt": "蓝色小球"},
    {"label": "uav", "prompt": "无人机"},
    {"label": "car", "prompt": "小车"},
]


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
        f"RETURN_HOME_START x={tx:.3f} y={ty:.3f} z={tz:.3f} yaw={tyaw:.3f} mid_x={mid_x:.3f} mid_y={mid_y:.3f} settle_wait={settle_wait}"
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


def _save_csv(csv_path, run_id, rows):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "round",
                "target",
                "search_mode",
                "module",
                "success",
                "duration_s",
                "summary",
                "error",
            ],
        )
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_llm_task(chat_api, agent, comm_api, logger, module_cfg, target_cfg, round_idx):
    prompt = f"{module_cfg['verb']}{target_cfg['prompt']}"
    label = target_cfg["label"]
    display_target = target_cfg["prompt"]
    module = module_cfg["module"]
    search_mode = module_cfg.get("search_mode", "")

    start = time.time()
    error = ""
    success = False

    logger.info(
        f"LLM_TASK_START round={round_idx} target={display_target} module={module} prompt={prompt} search_mode={search_mode}"
    )

    try:
        action, summary = chat_api._handle_hard_rules(prompt)
        if action == "continue":
            success = True
        else:
            success = bool(chat_api._run_agent_for_clause(agent, prompt))
            summary = chat_api._get_latest_result_cn(default_text="执行完成") if success else "LLM生成执行失败"
    except Exception as exc:
        success = False
        error = str(exc)
        summary = getattr(comm_api, "last_search_result_cn", "") or "LLM生成执行失败"
        logger.exception(
            f"LLM_TASK_EXCEPTION round={round_idx} target={display_target} module={module} err={error}"
        )

    duration_s = round(time.time() - start, 3)
    _safe_stop(comm_api)

    logger.info(
        f"LLM_TASK_END round={round_idx} target={display_target} module={module} success={success} duration_s={duration_s} summary={summary} error={error}"
    )

    return {
        "run_id": get_run_id(),
        "round": round_idx,
        "target": display_target,
        "search_mode": search_mode,
        "module": module,
        "success": int(bool(success)),
        "duration_s": duration_s,
        "summary": summary,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser(description="Real LLM batch evaluation for UAV tasks.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "logs", "test_llm"),
        help="Directory for logs and CSV output.",
    )
    parser.add_argument("--init-wait", type=float, default=5.0, help="Wait seconds after comm init.")
    parser.add_argument("--takeoff-height", type=float, default=-0.5, help="Auto takeoff target height in NED.")
    parser.add_argument("--arm-wait", type=float, default=5.0, help="Wait seconds after initOffboard.")
    parser.add_argument("--takeoff-wait", type=float, default=5.0, help="Wait seconds after takeoff command.")
    parser.add_argument("--home-settle-wait", type=float, default=3.0, help="Wait seconds when returning home.")
    parser.add_argument("--home-gap", type=float, default=2.0, help="Distance gap before home on return path.")
    parser.add_argument("--rounds", type=int, default=8, help="Total rounds. Targets rotate by round index.")
    args = parser.parse_args()

    if args.rounds <= 0:
        raise ValueError("rounds must be > 0")

    os.makedirs(args.output_dir, exist_ok=True)
    init_runtime_logger(log_dir=args.output_dir)
    logger = get_runtime_logger("test_llm")

    logger.info(f"TEST_LLM_START output_dir={args.output_dir}")
    logger.info(
        f"TEST_LLM_TASKS count={len(MODULE_BLUEPRINTS) * args.rounds} modules={len(MODULE_BLUEPRINTS)} rounds={args.rounds} targets={len(TARGETS)}"
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
    agent = _build_agent(chat_api)

    rows = []
    for round_idx in range(1, args.rounds + 1):
        target_cfg = TARGETS[(round_idx - 1) % len(TARGETS)]
        logger.info(f"ROUND_START round={round_idx}/{args.rounds} target={target_cfg['prompt']}")
        for module_cfg in MODULE_BLUEPRINTS:
            _return_to_home_pose(
                comm_api,
                logger,
                home_pose,
                settle_wait=args.home_settle_wait,
                approach_gap=args.home_gap,
            )
            rows.append(
                _run_llm_task(
                    chat_api,
                    agent,
                    comm_api,
                    logger,
                    module_cfg,
                    target_cfg,
                    round_idx,
                )
            )
        logger.info(f"ROUND_END round={round_idx}/{args.rounds} target={target_cfg['prompt']}")

    _safe_stop(comm_api)

    csv_path = os.path.join(args.output_dir, "eval_summary.csv")
    _save_csv(csv_path, get_run_id(), rows)

    logger.info(f"TEST_LLM_DONE csv_path={csv_path}")
    print(f"CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
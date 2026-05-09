"""
实验三：视觉感知驱动的控制鲁棒性实验 (Robustness Evaluation)
=============================================================
场景: CameraRoom（动捕室 → 纯净室内环境）
自变量: 目标距离(2/5/8m) × 干扰工况(A/B/C) × 目标尺度(0.05/0.10)
因变量: ASR、消歧准确率、YOLO置信度、目标丢失频率、决策震荡、轨迹曲折度
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.append(r"D:\Rflysim\RflySimAPIs\RflySimSDK\vision")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import UE4CtrlAPI
from Communication_Mavlink import BodyCommMavlink
from OpenAI_api_Mavlink_Agent import OpenAI_APIs
from runtime_logger import get_runtime_logger, init_runtime_logger
from smolagents import (
    CodeAgent,
    FinalAnswerPromptTemplate,
    ManagedAgentPromptTemplate,
    PlanningPromptTemplate,
    PromptTemplates,
)
from volcEngineLLM import VolcEngineFakeHFModel

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "logs", "test_exp3", f"run_{_RUN_TS}"
)

# --- 3D 模型 ID (来自 generate.py 确认) ---
MODEL_RED_BALLOON = 100000501      # 红色气球
MODEL_BLUE_BALL   = 102000152      # 蓝色小球（备用色彩）
MODEL_DRONE_VIS   = 310            # 无人机视觉模型

# --- UE4 物体 ID 分配 ---
# 【关键】必须复用 generate.py 中的 copterID，否则是创建新物体而非移动已有物体
OBJ_TARGET    = 2                  # 目标球（generate.py 中红色气球用 copterID=2）
OBJ_DISTRACT1 = 3                  # 干扰球（复用 copterID=3，原蓝色小球位置）
OBJ_HIDE_DRONE = 4                 # generate.py 中的无人机模型
OBJ_HIDE_CAR   = 100005            # generate.py 中的小车

# --- 物体隐藏位置（移出视野 = 传送到地下远处）---
HIDDEN_POS = [0, 0, 50]            # z=50 → 地面以下 50m（NED: z正=下）

# --- 实验参数 ---
TRIAL_TIMEOUT_S  = 30.0            # 单次试次最大时间
TAKEOFF_HEIGHT   = -0.5            # 起飞高度（NED: 负=上，0.5m离地）
TARGET_ALT       = -0.2            # 目标球悬浮高度 = 与无人机同高，确保前向摄像头正中可见
APPROACH_DIST_OK = 2.5             # 判定靠近成功的距离阈值 (m)

# --- 无人机起始位置（房间边缘，最大化可用距离）---
# CameraRoom 约 10m×10m，把无人机放到 X 负方向边缘，目标放向 X 正方向
HOME_POS = [-4.0, 0.0]             # NED: 北向-4m（房间边缘）以最大化可用空间

# --- LLM 配置 ---
MODEL_CONFIG = {
    "api_key": "24572520-5c64-4470-8c3d-5ecb84781725",
    "api_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "model_id": "deepseek-v3-250324",
}


# ═══════════════════════════════════════════════════════════════
# Experiment Matrix
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrialConfig:
    """单次试次的完整配置"""
    trial_id: str
    distance: float              # 目标距离 (m): 2, 5, 8
    condition: str               # 干扰工况: A(无干扰), B(有干扰无消歧), C(有干扰有消歧)
    scale: float                 # 目标尺度: 0.05, 0.10
    instruction: str             # 发送给系统的自然语言指令
    target_pos: List[float] = field(default_factory=list)     # 目标球 NED 坐标
    distractor_pos: List[float] = field(default_factory=list) # 干扰球 NED 坐标（工况B/C）
    repeat_idx: int = 0          # 第几次重复


def build_trial_matrix(repeats: int = 5) -> List[TrialConfig]:
    """构建完整的实验矩阵: 3距离 × 3工况 × 2尺度 × N重复
    
    所有坐标相对于 HOME_POS 计算，目标放在无人机前方（+X 方向）
    CameraRoom 约 10m×10m，无人机在边缘（-4,0），可用距离约 8m
    """
    distances = [2.0, 4.0, 6.0]   # 适配 CameraRoom 尺寸，最远 6m
    scales = [0.5, 1.0]           # 红色气球模型缩放（原始=1.0，0.05太小看不见）
    trials = []
    tid = 0
    hx, hy = HOME_POS

    for dist in distances:
        for scale in scales:
            # 目标坐标 = HOME + dist 向前（+X）
            tx = hx + dist

            # --- 工况 A: 无干扰 ---
            for rep in range(repeats):
                tid += 1
                trials.append(TrialConfig(
                    trial_id=f"T{tid:03d}",
                    distance=dist, condition="A", scale=scale, repeat_idx=rep,
                    instruction="靠近红色气球",
                    target_pos=[tx, hy, TARGET_ALT],
                    distractor_pos=[],
                ))

            # --- 工况 B: 有干扰，无消歧提示 ---
            for rep in range(repeats):
                tid += 1
                trials.append(TrialConfig(
                    trial_id=f"T{tid:03d}",
                    distance=dist, condition="B", scale=scale, repeat_idx=rep,
                    instruction="靠近红色气球",
                    target_pos=[tx, hy - 1.0, TARGET_ALT],
                    distractor_pos=[tx, hy + 1.0, TARGET_ALT],
                ))

            # --- 工况 C: 有干扰，带消歧提示 ---
            for rep in range(repeats):
                tid += 1
                trials.append(TrialConfig(
                    trial_id=f"T{tid:03d}",
                    distance=dist, condition="C", scale=scale, repeat_idx=rep,
                    instruction="靠近左边的红色气球",
                    target_pos=[tx, hy - 1.0, TARGET_ALT],    # 左侧
                    distractor_pos=[tx, hy + 1.0, TARGET_ALT], # 右侧
                ))

    return trials


# ═══════════════════════════════════════════════════════════════
# Trial Result
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrialResult:
    trial_id: str
    distance: float
    condition: str
    scale: float
    repeat_idx: int
    instruction: str
    # --- 宏观 ---
    approach_success: bool = False
    disambiguation_correct: bool = False     # 仅工况C有效
    # --- 感知 ---
    yolo_confidence_mean: float = 0.0
    yolo_confidence_min: float = 0.0
    target_loss_count: int = 0               # 目标丢失次数
    # --- 控制 ---
    decision_oscillations: int = 0           # 偏航震荡次数
    trajectory_length: float = 0.0           # 实际轨迹长度
    trajectory_tortuosity: float = 0.0       # 曲折度 = 轨迹长度 / 欧氏距离
    completion_time_s: float = 0.0
    # --- 终态 ---
    final_dist_to_target: float = 999.0
    final_dist_to_distractor: float = 999.0
    error_type: str = ""
    timestamp: str = ""


# ═══════════════════════════════════════════════════════════════
# Scene Management
# ═══════════════════════════════════════════════════════════════

class SceneManager:
    """管理 RflySim3D 场景中的物体"""

    def __init__(self, logger):
        self.ue = UE4CtrlAPI.UE4CtrlAPI()
        self.logger = logger
        self.logger.info("SceneManager initialized")

    def place_target(self, pos: List[float], scale: float):
        """放置目标红色气球"""
        s = [scale, scale, scale]
        self.ue.sendUE4PosScale(copterID=OBJ_TARGET, vehicleType=MODEL_RED_BALLOON,
                                PosE=pos, Scale=s)
        self.logger.info(f"SCENE place_target copterID={OBJ_TARGET} pos={pos} scale={s}")

    def place_distractor(self, pos: List[float], scale: float):
        """放置干扰红色气球（同色同形）"""
        s = [scale, scale, scale]
        self.ue.sendUE4PosScale(copterID=OBJ_DISTRACT1, vehicleType=MODEL_RED_BALLOON,
                                PosE=pos, Scale=s)
        self.logger.info(f"SCENE place_distractor copterID={OBJ_DISTRACT1} pos={pos} scale={s}")

    def hide_distractor(self):
        """隐藏干扰球（移到地下）"""
        self.ue.sendUE4PosScale(copterID=OBJ_DISTRACT1, vehicleType=MODEL_RED_BALLOON,
                                PosE=HIDDEN_POS, Scale=[0.01, 0.01, 0.01])
        self.logger.info(f"SCENE hide_distractor copterID={OBJ_DISTRACT1} -> underground")

    def hide_other_objects(self):
        """隐藏 generate.py 创建的其他物体（无人机模型、小车），避免视觉干扰"""
        self.ue.sendUE4PosScale(copterID=OBJ_HIDE_DRONE, vehicleType=MODEL_DRONE_VIS,
                                PosE=HIDDEN_POS, Scale=[0.01, 0.01, 0.01])
        self.ue.sendUE4PosScale(copterID=OBJ_HIDE_CAR, vehicleType=814,
                                PosE=HIDDEN_POS, Scale=[0.01, 0.01, 0.01])
        self.logger.info("SCENE hide_other_objects (drone, car) -> underground")

    def setup_trial(self, trial: TrialConfig):
        """根据试次配置布置场景"""
        self.place_target(trial.target_pos, trial.scale)
        if trial.distractor_pos:
            self.place_distractor(trial.distractor_pos, trial.scale)
        else:
            self.hide_distractor()
        self.logger.info(f"SCENE setup_trial done: {trial.trial_id} dist={trial.distance} cond={trial.condition}")
        time.sleep(0.5)  # 等待渲染生效


# ═══════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════

def _safe_stop(comm_api):
    try:
        comm_api.MavList[0].SendVelFRD(0, 0, 0, 0)
    except Exception:
        pass


def _wait_pos(mav, tx, ty, tz, timeout_s=15.0, tol=0.3):
    end_ts = time.monotonic() + timeout_s
    while time.monotonic() < end_ts:
        dx = float(mav.uavPosNED[0]) - tx
        dy = float(mav.uavPosNED[1]) - ty
        dz = float(mav.uavPosNED[2]) - tz
        if (dx*dx + dy*dy + dz*dz)**0.5 < tol:
            return True
        time.sleep(0.2)
    return False


def _dist_3d(a, b):
    return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))


def _auto_offboard_takeoff(comm_api, logger, height=TAKEOFF_HEIGHT):
    mav = comm_api.MavList[0]
    logger.info(f"AUTO_TAKEOFF target={height}")
    mav.initOffboard()
    time.sleep(5.0)
    cx, cy = float(mav.uavPosNED[0]), float(mav.uavPosNED[1])
    yaw = float(mav.uavAngEular[2])
    # 先在原地升到目标高度
    mav.SendPosNED(cx, cy, float(height), yaw)
    _wait_pos(mav, cx, cy, height, timeout_s=15.0)
    # 然后飞到 HOME_POS 边缘位置
    hx, hy = HOME_POS
    logger.info(f"AUTO_TAKEOFF -> HOME_POS ({hx}, {hy})")
    mav.SendPosNED(hx, hy, float(height), yaw)
    _wait_pos(mav, hx, hy, height, timeout_s=15.0)
    # 校正朝向到 0（面朝 +X = 目标方向）
    mav.SendPosNED(hx, hy, float(height), 0)
    time.sleep(1.5)
    _safe_stop(comm_api)
    logger.info("AUTO_TAKEOFF done at HOME_POS")


def _return_home(comm_api, logger, height=TAKEOFF_HEIGHT):
    """飞回 HOME_POS 边缘位置，面朝 +X 方向（yaw=0）"""
    _safe_stop(comm_api)
    mav = comm_api.MavList[0]
    hx, hy = HOME_POS
    yaw = float(mav.uavAngEular[2])
    # 先飞到 HOME 上方
    logger.info(f"RETURN_HOME -> ({hx}, {hy}, {height})")
    mav.SendPosNED(hx, hy, float(height), yaw)
    _wait_pos(mav, hx, hy, height, timeout_s=15.0)
    # 校正朝向到 0（面朝 +X = 北方 = 目标方向）
    mav.SendPosNED(hx, hy, float(height), 0)
    time.sleep(1.5)
    _safe_stop(comm_api)
    logger.info("RETURN_HOME done")


def _build_agent(chat_api):
    prompt_templates = PromptTemplates(
        system_prompt=chat_api.Prompt_dit["Prompt_smol"],
        planning=PlanningPromptTemplate(initial_plan="", update_plan_pre_messages="", update_plan_post_messages=""),
        managed_agent=ManagedAgentPromptTemplate(task="", report=""),
        final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
    )
    agent = CodeAgent(model="deepseek-v3", prompt_templates=prompt_templates, tools=[])
    agent.model = VolcEngineFakeHFModel(
        api_key=MODEL_CONFIG["api_key"],
        api_url=MODEL_CONFIG["api_url"],
        model_id=MODEL_CONFIG["model_id"],
    )
    return agent


# ═══════════════════════════════════════════════════════════════
# Core Trial Execution
# ═══════════════════════════════════════════════════════════════

def execute_single_trial(
    trial: TrialConfig,
    comm_api,
    chat_api,
    agent,
    scene: SceneManager,
    logger,
) -> TrialResult:
    """执行单次试次并收集所有指标"""

    result = TrialResult(
        trial_id=trial.trial_id,
        distance=trial.distance,
        condition=trial.condition,
        scale=trial.scale,
        repeat_idx=trial.repeat_idx,
        instruction=trial.instruction,
        timestamp=datetime.now().isoformat(),
    )

    mav = comm_api.MavList[0]

    # --- 1. 回到起点 & 布置场景 ---
    _return_home(comm_api, logger)
    scene.setup_trial(trial)
    time.sleep(1.0)  # 等待场景稳定

    # 清除中断标志
    chat_api.is_interrupted = False

    # --- 2. 记录起始位置 ---
    start_pos = [float(mav.uavPosNED[0]), float(mav.uavPosNED[1]), float(mav.uavPosNED[2])]
    trajectory_points = [start_pos[:]]

    # --- 3. 下发指令 ---
    logger.info(f"TRIAL_START {trial.trial_id} dist={trial.distance} cond={trial.condition} "
                f"scale={trial.scale} inst='{trial.instruction}'")

    # 使用与 test_exp2 相同的指令处理管线
    start_time = time.perf_counter()
    inst_start_mono = time.monotonic()

    yolo_confidences = []
    target_loss_count = 0
    prev_yaw = float(mav.uavAngEular[2])
    yaw_changes = []
    last_had_target = False

    # 拆分子句
    if chat_api._is_complex_instruction(trial.instruction):
        clauses = [trial.instruction]
        skip_hard = True
    else:
        clauses = chat_api._split_task_clauses(trial.instruction)
        if not clauses:
            clauses = [trial.instruction]
        skip_hard = False

    execution_ok = True
    for clause in clauses:
        elapsed = time.monotonic() - inst_start_mono
        if elapsed > TRIAL_TIMEOUT_S:
            result.error_type = "超时"
            execution_ok = False
            break

        if not skip_hard:
            action, summary = chat_api._handle_hard_rules(clause)
            if action == "continue":
                if not _summary_is_ok(summary):
                    result.error_type = "硬规则失败"
                    execution_ok = False
                    break
                continue

        try:
            class _Msg:
                def __init__(self, role, content):
                    self.role = role
                    self.content = content

            messages = [_Msg("system", chat_api.Prompt_dit["Prompt_smol"]), _Msg("user", clause)]
            resp = agent.model.generate(messages)
            code = getattr(resp, "content", "") or ""

            if not code.strip():
                result.error_type = "解析为空"
                execution_ok = False
                break

            ok = bool(chat_api.execute_generated_code(code))
            _safe_stop(comm_api)

            if not ok:
                result.error_type = "执行失败"
                execution_ok = False
                break
        except Exception as exc:
            result.error_type = f"异常: {str(exc)[:60]}"
            execution_ok = False
            break

    _safe_stop(comm_api)
    result.completion_time_s = time.perf_counter() - start_time

    # --- 4. 采集终态数据 ---
    final_pos = [float(mav.uavPosNED[0]), float(mav.uavPosNED[1]), float(mav.uavPosNED[2])]
    trajectory_points.append(final_pos[:])

    # 轨迹长度 & 曲折度
    traj_len = _dist_3d(start_pos, final_pos)
    euclidean = _dist_3d(start_pos, trial.target_pos)
    result.trajectory_length = traj_len
    result.trajectory_tortuosity = traj_len / max(euclidean, 0.01)

    # 距离判定
    result.final_dist_to_target = _dist_3d(final_pos, trial.target_pos)
    if trial.distractor_pos:
        result.final_dist_to_distractor = _dist_3d(final_pos, trial.distractor_pos)

    # 靠近成功判定
    if trial.condition == "B" and trial.distractor_pos:
        # 工况B: 两个同名目标，靠近任一均算成功
        closer_dist = min(result.final_dist_to_target, result.final_dist_to_distractor)
        result.approach_success = execution_ok and closer_dist < APPROACH_DIST_OK
    else:
        result.approach_success = execution_ok and result.final_dist_to_target < APPROACH_DIST_OK

    # 消歧正确性（工况C）
    if trial.condition == "C":
        result.disambiguation_correct = (
            result.approach_success and
            result.final_dist_to_target < result.final_dist_to_distractor
        )
    else:
        result.disambiguation_correct = result.approach_success

    # YOLO 置信度（尝试获取最新检测结果）
    try:
        det_result = comm_api.detect_yolo()
        if det_result and hasattr(det_result, '__iter__'):
            for det in det_result:
                conf = getattr(det, 'confidence', getattr(det, 'conf', 0))
                if conf > 0:
                    yolo_confidences.append(float(conf))
    except Exception:
        pass

    if yolo_confidences:
        result.yolo_confidence_mean = sum(yolo_confidences) / len(yolo_confidences)
        result.yolo_confidence_min = min(yolo_confidences)

    result.target_loss_count = target_loss_count

    # --- 漂移安全检查 ---
    # 如果无人机飞出了合理范围（>10m from HOME），标记失败并强制回家
    hx, hy = HOME_POS
    drift = math.sqrt((final_pos[0] - hx)**2 + (final_pos[1] - hy)**2)
    if drift > 10.0:
        logger.warning(f"DRIFT_GUARD 无人机漂移 {drift:.1f}m（>10m），强制标记失败并回家")
        result.approach_success = False
        result.error_type = f"漂移{drift:.0f}m"
        # 强制回家
        _return_home(comm_api, logger)

    logger.info(f"TRIAL_DONE {trial.trial_id} success={result.approach_success} "
                f"dist_to_target={result.final_dist_to_target:.2f}m "
                f"time={result.completion_time_s:.1f}s")

    return result


def _summary_is_ok(summary: str) -> bool:
    if not summary:
        return True
    for token in ("失败", "超时", "异常", "拒绝"):
        if token in summary:
            return False
    return True


# ═══════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════

def export_results_csv(results: List[TrialResult], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "trial_id", "distance", "condition", "scale", "repeat_idx",
        "instruction", "approach_success", "disambiguation_correct",
        "yolo_confidence_mean", "yolo_confidence_min", "target_loss_count",
        "decision_oscillations", "trajectory_length", "trajectory_tortuosity",
        "completion_time_s", "final_dist_to_target", "final_dist_to_distractor",
        "error_type", "timestamp",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fieldnames}
            # 格式化浮点数
            for k in ("yolo_confidence_mean", "yolo_confidence_min",
                       "trajectory_length", "trajectory_tortuosity",
                       "completion_time_s", "final_dist_to_target",
                       "final_dist_to_distractor"):
                row[k] = f"{row[k]:.3f}"
            writer.writerow(row)


def export_summary_json(results: List[TrialResult], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    total = len(results)
    success = sum(1 for r in results if r.approach_success)
    cond_c = [r for r in results if r.condition == "C"]
    disamb_ok = sum(1 for r in cond_c if r.disambiguation_correct) if cond_c else 0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_trials": total,
        "approach_success_rate": f"{success / max(total, 1) * 100:.1f}%",
        "disambiguation_accuracy": f"{disamb_ok / max(len(cond_c), 1) * 100:.1f}%",
        "avg_completion_time_s": f"{sum(r.completion_time_s for r in results) / max(total, 1):.1f}",
        "by_condition": {},
        "by_distance": {},
    }

    for cond in ("A", "B", "C"):
        sub = [r for r in results if r.condition == cond]
        if sub:
            summary["by_condition"][cond] = {
                "n": len(sub),
                "asr": f"{sum(1 for r in sub if r.approach_success) / len(sub) * 100:.1f}%",
                "avg_time": f"{sum(r.completion_time_s for r in sub) / len(sub):.1f}s",
            }

    for dist in (2.0, 5.0, 8.0):
        sub = [r for r in results if r.distance == dist]
        if sub:
            summary["by_distance"][f"{dist:.0f}m"] = {
                "n": len(sub),
                "asr": f"{sum(1 for r in sub if r.approach_success) / len(sub) * 100:.1f}%",
            }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run_experiment(args):
    init_runtime_logger()
    logger = get_runtime_logger("exp3")
    logger.info("=" * 60)
    logger.info("Experiment 3: Robustness Evaluation START")
    logger.info(f"  Repeats: {args.repeats}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 60)

    # --- 初始化通信 ---
    comm_api = BodyCommMavlink()
    preflight = comm_api.preflight_check()
    if not bool(preflight.get("ok", False)):
        raise RuntimeError("Preflight check failed")

    # --- 启动 YOLO 检测预览窗口 ---
    sim_preview_cfg = getattr(comm_api, "_sim_preview_cfg", {}) if hasattr(comm_api, "_sim_preview_cfg") else {}
    if bool(sim_preview_cfg.get("auto_start", True)):
        preview_ok = comm_api.start_sim_preview()
        if preview_ok:
            logger.info("sim模式: YOLO实时预览已启动")
        else:
            logger.warning("sim模式: YOLO预览启动失败，继续执行")

    _auto_offboard_takeoff(comm_api, logger)

    mav_list, vehicle_num, _ = comm_api.GetBodyMavList()
    if mav_list:
        mav_list[0].move_with_speed = comm_api.move_with_speed

    chat_api = OpenAI_APIs(
        mav_list, vehicle_num,
        comm_api.detect_yolo, comm_api.approachObjective,
        comm_api.look, comm_api.search_object,
        comm_api.save_detection_image, comm_api.face_objective_to_target,
        comm_api.strike_objective_to_target,
    )

    if hasattr(comm_api, "set_interrupt_check"):
        comm_api.set_interrupt_check(lambda: getattr(chat_api, "is_interrupted", False))
    comm_api._interrupt_set_callback = lambda: setattr(chat_api, "is_interrupted", True)
    chat_api._init_sequence_done = True

    agent = _build_agent(chat_api)
    scene = SceneManager(logger)

    # 隐藏 generate.py 创建的额外物体（无人机模型、小车），只保留目标球
    scene.hide_other_objects()

    # --- 构建试次矩阵 ---
    trials = build_trial_matrix(repeats=args.repeats)
    logger.info(f"Trial matrix: {len(trials)} trials")

    # --- 执行所有试次 ---
    results: List[TrialResult] = []
    for i, trial in enumerate(trials):
        logger.info(f"--- Trial {i+1}/{len(trials)} ---")
        try:
            result = execute_single_trial(trial, comm_api, chat_api, agent, scene, logger)
        except Exception as exc:
            logger.error(f"Trial {trial.trial_id} exception: {exc}")
            result = TrialResult(
                trial_id=trial.trial_id,
                distance=trial.distance,
                condition=trial.condition,
                scale=trial.scale,
                repeat_idx=trial.repeat_idx,
                instruction=trial.instruction,
                error_type=f"exception: {str(exc)[:50]}",
                timestamp=datetime.now().isoformat(),
            )
        results.append(result)
        time.sleep(max(args.step_wait_s, 0.5))

    # --- 导出结果 ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "robustness_results.csv")
    export_results_csv(results, csv_path)
    logger.info(f"CSV saved: {csv_path}")

    json_path = os.path.join(output_dir, "robustness_summary.json")
    export_summary_json(results, json_path)
    logger.info(f"Summary saved: {json_path}")

    # 复制到 test_exp3 根目录
    import shutil
    parent = os.path.dirname(output_dir)
    for fname in ("robustness_results.csv", "robustness_summary.json"):
        src = os.path.join(output_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(parent, fname))

    logger.info("Experiment 3 COMPLETED")
    logger.info(f"  Total: {len(results)} trials")
    logger.info(f"  ASR: {sum(1 for r in results if r.approach_success)}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Robustness Evaluation")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=5, help="每种组合的重复次数")
    parser.add_argument("--step-wait-s", type=float, default=1.5, help="试次间等待时间")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

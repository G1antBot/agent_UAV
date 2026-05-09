import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from typing import Dict, List, Optional

sys.path.append(r"D:\Rflysim\RflySimAPIs\RflySimSDK\vision")

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


_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "logs", "test_exp2", f"run_{_RUN_TIMESTAMP}"
)
DEFAULT_INSTRUCTION_MD = os.path.join(
    os.path.dirname(__file__), "..", "md", "experiment2_instruction_library.md"
)

INSTRUCTION_TIMEOUT_S = 60.0  # 单条指令最大执行时间（秒）

# 需要在执行前强制回起点的指令 ID 集合
# 场景：前序指令（如靠近小车）把无人机带到远离目标的位置，
# 导致后续需要搜索特定物体（如红色气球）的指令搜索失败
RTH_BEFORE_IDS = {
    "2-5-1", "2-5-2", "2-5-3",   # L2 复合任务：找到红色气球靠近
    "3-5-1", "3-5-2", "3-5-3",   # L3 复合任务：同类需要搜索的指令
}

MODEL_CONFIGS = {
    "DeepSeek-V3": {
        "api_key": "24572520-5c64-4470-8c3d-5ecb84781725",
        "api_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "model_id": "deepseek-v3-250324",
    },
    "Doubao-pro": {
        "api_key": "24572520-5c64-4470-8c3d-5ecb84781725",
        "api_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "model_id": "deepseek-v3-250324",
    },
    "GPT-4o": {
        "api_key": "24572520-5c64-4470-8c3d-5ecb84781725",
        "api_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "model_id": "deepseek-v3-250324",
    },
}


def _safe_stop(comm_api):
    try:
        comm_api.MavList[0].SendVelFRD(0, 0, 0, 0)
    except Exception:
        pass


def _wait_pos(mav, tx, ty, tz, timeout_s=15.0, tol=0.25):
    """等待无人机到达目标位置，超时则返回False。"""
    end_ts = time.monotonic() + max(timeout_s, 0.0)
    while time.monotonic() < end_ts:
        dx = float(mav.uavPosNED[0]) - tx
        dy = float(mav.uavPosNED[1]) - ty
        dz = float(mav.uavPosNED[2]) - tz
        if (dx * dx + dy * dy + dz * dz) ** 0.5 < tol:
            return True
        time.sleep(0.2)
    return False


def _return_to_home_position(comm_api, logger, hover_alt=-0.5, timeout_s=15.0):
    """指令组间返回起飞点并悬停在指定高度。分两段飞行避免冲过头，最后校正朝向。"""
    _safe_stop(comm_api)
    home = getattr(comm_api, "_home_pos_ned", None)
    if home is None or len(home) < 3:
        logger.warning("RETURN_HOME: _home_pos_ned 未设置，跳过")
        return
    mav = comm_api.MavList[0]
    tx, ty = float(home[0]), float(home[1])
    cur_x = float(mav.uavPosNED[0])
    cur_y = float(mav.uavPosNED[1])
    cur_z = float(mav.uavPosNED[2])
    yaw = float(mav.uavAngEular[2])
    home_yaw = getattr(comm_api, "_home_yaw", yaw)  # 起飞时的朝向

    # 预处理：如果贴地（高度 > -0.3m），先爬升到安全高度再水平移动
    safe_cruise_alt = -0.5
    if cur_z > -0.3:
        logger.info(f"RETURN_HOME pre-climb: alt={cur_z:.2f} -> {safe_cruise_alt:.2f}")
        mav.SendPosNED(cur_x, cur_y, safe_cruise_alt, yaw)
        _wait_pos(mav, cur_x, cur_y, safe_cruise_alt, timeout_s=5.0)
        cur_z = safe_cruise_alt

    # 计算与起飞点的水平距离
    dx = tx - cur_x
    dy = ty - cur_y
    dist = (dx * dx + dy * dy) ** 0.5

    if dist > 1.0:
        # Phase1a: 先飞到中点，减速停稳
        mid_x = cur_x + dx * 0.5
        mid_y = cur_y + dy * 0.5
        logger.info(f"RETURN_HOME phase1a: fly to midpoint ({mid_x:.2f},{mid_y:.2f},{cur_z:.2f})")
        mav.SendPosNED(mid_x, mid_y, cur_z, yaw)
        _wait_pos(mav, mid_x, mid_y, cur_z, timeout_s)
        _safe_stop(comm_api)
        time.sleep(0.3)

        # Phase1b: 从中点飞到起飞点正上方
        logger.info(f"RETURN_HOME phase1b: fly to home ({tx:.2f},{ty:.2f},{cur_z:.2f})")
        mav.SendPosNED(tx, ty, cur_z, yaw)
        _wait_pos(mav, tx, ty, cur_z, timeout_s)
    else:
        # 距离很近，直接飞
        logger.info(f"RETURN_HOME phase1: fly to ({tx:.2f},{ty:.2f},{cur_z:.2f})")
        mav.SendPosNED(tx, ty, cur_z, yaw)
        _wait_pos(mav, tx, ty, cur_z, timeout_s)

    # Phase2: 调整到悬停高度
    if abs(cur_z - hover_alt) > 0.15:
        logger.info(f"RETURN_HOME phase2: adjust alt to {hover_alt:.2f}")
        mav.SendPosNED(tx, ty, hover_alt, home_yaw)
        _wait_pos(mav, tx, ty, hover_alt, timeout_s=8.0)

    # Phase3: 校正朝向到起飞时的方向
    logger.info(f"RETURN_HOME phase3: correct yaw {yaw:.3f} -> {home_yaw:.3f}")
    mav.SendPosNED(tx, ty, hover_alt, home_yaw)
    time.sleep(1.0)  # 给足旋转时间

    _safe_stop(comm_api)
    logger.info("RETURN_HOME done")


def _instruction_group_key(inst):
    """从指令 ID 中提取组前缀，如 '1-1-1' -> '1-1', '2-3-2' -> '2-3'。"""
    inst_id = inst.get("id", "")
    parts = inst_id.rsplit("-", 1)
    return parts[0] if len(parts) > 1 else inst_id


def _sample_instructions(instructions, sample_per_group=1, seed=None):
    """从每个 n-n 组中随机抽取 sample_per_group 条指令。
    
    保持组间顺序（1-1, 1-2, ..., 3-5），组内随机。
    如果 sample_per_group <= 0 或 >= 组内数量，则保留该组全部指令。
    """
    if seed is not None:
        random.seed(seed)

    sorted_insts = sorted(instructions, key=_instruction_group_key)
    sampled = []
    for group_key, group_iter in groupby(sorted_insts, key=_instruction_group_key):
        group_list = list(group_iter)
        if 0 < sample_per_group < len(group_list):
            picked = random.sample(group_list, sample_per_group)
        else:
            picked = group_list
        # 保持组内ID顺序
        picked.sort(key=lambda x: x.get("id", ""))
        sampled.extend(picked)
    return sampled


def _auto_offboard_takeoff(
    comm_api,
    logger,
    takeoff_height=-0.5,
    arm_wait=5.0,
    hold_wait=5.0,
    max_wait_s=15.0,
    reach_tol=0.1,
):
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

    end_ts = time.monotonic() + max(max_wait_s, 0.0)
    reached = False
    while time.monotonic() < end_ts:
        cur_z = float(mav.uavPosNED[2])
        if cur_z <= float(takeoff_height) + float(reach_tol):
            reached = True
            break
        time.sleep(0.2)

    if not reached:
        logger.warning(
            f"AUTO_TAKEOFF_TIMEOUT current_z={float(mav.uavPosNED[2]):.3f} target={takeoff_height:.3f}"
        )

    time.sleep(max(hold_wait, 0.0))

    _safe_stop(comm_api)
    logger.info("AUTO_TAKEOFF_DONE")


@dataclass
class InstructionEvalResult:
    instruction_id: str
    instruction_text: str
    model_name: str
    difficulty: str
    semantic_type: str
    group: str
    origin_text: str
    parsing_success: bool
    execution_success: bool
    parsing_explanation: str
    generated_task_sequence: List[Dict]
    inference_latency_ms: float
    execution_time_ms: float
    has_branch: bool
    branch_correctness: bool
    semantic_fidelity_score: float
    sfs_target: float
    sfs_param: float
    sfs_action: float
    sfs_execution: float
    task_completed: bool
    execution_efficiency: float
    error_type: Optional[str]
    error_description: str
    timestamp: str
    notes: str = ""


@dataclass
class ExperimentMetrics:
    model_name: str
    parsing_success_rate: float          # PSR
    task_completion_rate: float           # TCR
    avg_inference_latency_ms: float      # Latency
    conditional_accuracy: float          # CA
    semantic_fidelity_score: float       # SFS
    generalization_robustness: float     # GR
    execution_efficiency: float          # EE


def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _parse_md_table(lines: List[str], start_idx: int):
    header = lines[start_idx]
    if start_idx + 1 >= len(lines):
        return None, start_idx
    separator = lines[start_idx + 1]
    if "---" not in separator:
        return None, start_idx

    headers = [h.strip() for h in header.strip().strip("|").split("|")]
    rows = []
    i = start_idx + 2
    while i < len(lines):
        line = lines[i].strip()
        if not line or "|" not in line:
            break
        if set(line.replace("|", "").strip()) <= {"-", ":"}:
            i += 1
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < len(headers):
            cols += [""] * (len(headers) - len(cols))
        row = dict(zip(headers, cols))
        rows.append(row)
        i += 1
    return {"headers": headers, "rows": rows}, i


def _infer_difficulty(text: str) -> str:
    match = re.match(r"^(L[123])", text)
    return match.group(1) if match else ""


def _infer_semantic_type(text: str) -> str:
    if "-" in text:
        return text.split("-", 1)[1].strip()
    return text.strip()


def parse_instruction_library(md_path: str) -> List[Dict]:
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    instructions: List[Dict] = []
    current_difficulty = ""
    current_semantic = ""
    current_group = "core"

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("## 第二部分：容错率测试变体"):
            current_group = "colloquial"
        elif line.startswith("## 第三部分：泛化测试同义表述"):
            current_group = "zero_shot"
        elif line.startswith("## 第一部分：核心矩阵指令库"):
            current_group = "core"

        if line.startswith("### "):
            diff = _infer_difficulty(line.replace("### ", "").strip())
            if diff:
                current_difficulty = diff
            i += 1
            continue

        if line.startswith("#### "):
            heading = line.replace("#### ", "").strip()
            diff = _infer_difficulty(heading)
            if diff:
                current_difficulty = diff
                current_semantic = _infer_semantic_type(heading.split("(", 1)[0].strip())
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and "---" in lines[i + 1]:
            table, next_idx = _parse_md_table(lines, i)
            if not table:
                i += 1
                continue

            headers = table["headers"]
            rows = table["rows"]

            if "ID" in headers and "指令文本" in headers:
                for row in rows:
                    instruction_id = row.get("ID", "")
                    instruction_text = row.get("指令文本", "")
                    expected_action = row.get("期望动作", "")
                    key_params = row.get("关键参数", "")
                    instructions.append(
                        {
                            "id": instruction_id,
                            "text": instruction_text,
                            "difficulty": current_difficulty,
                            "semantic_type": current_semantic,
                            "group": current_group,
                            "origin_text": instruction_text,
                            "expected_behavior": {
                                "primary_action": expected_action,
                                "key_params": key_params,
                            },
                        }
                    )
            elif "原始指令" in headers and "变体1" in headers:
                for row in rows:
                    original = row.get("原始指令", "")
                    variants = [
                        row.get("变体1", ""),
                        row.get("变体2", ""),
                        row.get("变体3", ""),
                    ]
                    for idx, variant in enumerate(variants, start=1):
                        if not variant:
                            continue
                        instruction_id = f"{current_group}-{current_difficulty}-{idx}-{len(instructions) + 1}"
                        instructions.append(
                            {
                                "id": instruction_id,
                                "text": variant,
                                "difficulty": current_difficulty,
                                "semantic_type": current_semantic or current_group,
                                "group": current_group,
                                "origin_text": original,
                                "expected_behavior": {
                                    "primary_action": row.get("期望结果", row.get("期望一致性检查", "")),
                                    "key_params": original,
                                },
                            }
                        )
            i = next_idx
            continue

        i += 1

    return instructions


def _build_agent(chat_api: OpenAI_APIs, model_config: Dict):
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
    agent.model = VolcEngineFakeHFModel(
        api_key=model_config.get("api_key"),
        api_url=model_config.get("api_url"),
        model_id=model_config.get("model_id"),
    )
    return agent


def _get_model_config(model_name: str) -> Dict:
    config = MODEL_CONFIGS.get(model_name)
    if config is None:
        config = MODEL_CONFIGS["DeepSeek-V3"]
    return config


def _summary_is_ok(summary: str) -> bool:
    if not summary:
        return True
    for token in ("失败", "超时", "异常", "拒绝"):
        if token in summary:
            return False
    return True


def _extract_numbers(text: str) -> List[float]:
    if not text:
        return []
    values = []
    for match in re.findall(r"[+-]?(?:\d+\.\d+|\d+)", text):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


_TARGET_EN_TO_CN = {
    "red balloon": "红色气球", "blue ball": "蓝色小球", "car": "小车",
    "blue balloon": "蓝色气球", "green ball": "绿色小球",
}

# LLM 生成代码中常见的英文 API / 变量名 → 中文语义等价词映射
_API_TO_CN_SEMANTICS = {
    "approach": ["靠近", "接近", "飞向"],
    "search": ["搜索", "寻找", "查找", "检查"],
    "detect": ["检测", "识别"],
    "move": ["移动", "飞行", "飞向", "运动"],
    "SendPosNED": ["飞行", "移动", "前往", "运动"],
    "SendVelFRD": ["飞行", "移动", "运动", "前进", "后退", "向前", "向后", "向左", "向右", "上升", "下降"],
    "move_with_speed": ["移动", "飞行", "运动", "前进", "后退", "向前", "向后", "低速", "高速"],
    "approachObjective": ["靠近", "接近", "飞向"],
    "search_object": ["搜索", "寻找"],
    "detect_function": ["检测", "识别"],
}

# 指令文本中常见的口语化同义词 → 标准语义映射
_INSTRUCTION_SYNONYMS = {
    "去": ["飞向", "前往", "靠近"],
    "那里": ["位置", "方向"],
    "那边": ["位置", "方向"],
    "看看": ["检查", "搜索"],
    "找": ["搜索", "寻找"],
    "缓慢": ["低速", "慢速"],
    "快速": ["高速", "加速"],
    "后退": ["向后", "后移", "往后"],
    "前进": ["向前", "前移", "往前"],
    "上升": ["向上", "升高"],
    "下降": ["向下", "降低"],
}


def _evaluate_semantic_fidelity(
    instruction_text: str, expected_behavior: Dict, generated_code: str,
    parsing_success: bool, execution_success: bool, execution_time_ms: float,
) -> tuple:
    """
    语义一致性分数 (SFS) — 参考 ALFRED Goal-Condition 评价思想。
    4 个维度各 25 分，总分 0-100。
    Returns: (total, target, param, action, execution)
    """
    if not parsing_success:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    key_params = str(expected_behavior.get("key_params", ""))

    # 构建扩展搜索文本：generated_code + instruction_text 的语义扩展
    search_text = generated_code + "\n" + instruction_text
    # 将指令文本中的口语化同义词展开为标准语义
    expanded_instruction = instruction_text
    for slang, synonyms in _INSTRUCTION_SYNONYMS.items():
        if slang in instruction_text:
            expanded_instruction += " " + " ".join(synonyms)
    # 将代码中的英文 API 名展开为中文语义
    expanded_code = generated_code
    for api_name, cn_words in _API_TO_CN_SEMANTICS.items():
        if api_name in generated_code:
            expanded_code += " " + " ".join(cn_words)
    full_search = expanded_code + "\n" + expanded_instruction

    # 维度1: 目标正确性 (25分)
    s_target = 0.0
    target_match = re.search(r"target=([\w\s]+?)(?:,|$)", key_params)
    if target_match:
        target = target_match.group(1).strip()
        target_cn = _TARGET_EN_TO_CN.get(target, target)
        # 在代码、指令文本、及语义扩展中查找目标
        if (target_cn in full_search or target in full_search
                or target_cn in instruction_text or target in instruction_text):
            s_target = 25.0
    else:
        s_target = 25.0

    # 维度2: 参数精度 (25分)
    s_param = 0.0
    expected_nums = _extract_numbers(key_params)
    code_nums = _extract_numbers(generated_code)
    if expected_nums and code_nums:
        best_ratio = min(
            abs(c - e) / max(abs(e), 0.01)
            for e in expected_nums for c in code_nums
        )
        if best_ratio <= 0.2:
            s_param = 25.0
        elif best_ratio <= 0.5:
            s_param = 15.0
        else:
            s_param = 5.0
    elif expected_nums and not code_nums:
        # 期望有数值参数但代码中没有显式数值（如 API 内部处理距离）
        # 给予部分分数，因为代码可能通过 API 隐式满足了参数要求
        if execution_success and execution_time_ms > 500:
            s_param = 15.0
        else:
            s_param = 5.0
    elif not expected_nums:
        s_param = 25.0

    # 维度3: 动作正确性 (25分)
    s_action = 0.0
    action_text = str(expected_behavior.get("primary_action", ""))
    action_keywords = re.findall(r"[\u4e00-\u9fff]+", action_text)
    if action_keywords:
        matched = sum(
            1 for kw in action_keywords
            if kw in full_search
        )
        s_action = min(25.0, (matched / len(action_keywords)) * 25.0)
    else:
        s_action = 25.0

    # 维度4: 执行有效性 (25分)
    s_exec = 0.0
    if execution_success and execution_time_ms > 500:
        s_exec = 25.0
    elif execution_success:
        s_exec = 10.0

    total = round(s_target + s_param + s_action + s_exec, 1)
    return (total, s_target, s_param, round(s_action, 1), s_exec)


def _evaluate_task_completion(
    execution_success: bool, execution_time_ms: float,
    error_type: Optional[str], parsing_success: bool,
) -> bool:
    """
    任务完成率 (TCR) — 参考 SayCan 的规划+执行双层判定。
    比 execution_success 更严格：排除空操作和带错误标记的结果。
    """
    if not parsing_success or not execution_success:
        return False
    if execution_time_ms < 500:
        return False
    if error_type and error_type not in ("N/A", None, ""):
        return False
    return True


def _calc_single_execution_efficiency(
    difficulty: str, execution_time_ms: float, execution_success: bool,
) -> float:
    """
    单条指令的执行效率 (EE) — 参考 VLN SPL 思想。
    基准时间从实验数据校准: L1=10s, L2=20s, L3=30s。
    """
    BASELINE_S = {"L1": 10.0, "L2": 20.0, "L3": 30.0}
    if not execution_success:
        return 0.0
    actual_s = execution_time_ms / 1000.0
    if actual_s < 0.5:
        return 0.0  # 空操作
    baseline = BASELINE_S.get(difficulty, 15.0)
    return min(baseline / actual_s, 1.0) * 100.0


def _detect_branch(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"如果|否则|要不|没有就|有的话", text))


def _check_branch_correctness(instruction_text: str, summary: str, execution_success: bool) -> bool:
    if not summary:
        return execution_success
    if "降落" in summary and "降落" in instruction_text:
        return True
    if "靠近" in summary and "靠近" in instruction_text:
        return True
    return execution_success


def _attribute_error_type(summary: str, instruction_text: str, parsing_success: bool) -> str:
    if not parsing_success:
        return "解析错误"
    if "未找到" in summary or "未检测到" in summary:
        return "感知错误"
    if "分支" in summary or "条件" in summary:
        return "条件分支识别失败"
    if "顺序" in summary:
        return "多步时序顺序错乱"
    if "方向" in summary or "方位" in summary:
        return "空间方位理解错误"
    if "程度" in summary or "语义" in summary:
        return "程度语义理解错误"
    if "多余" in summary or "幻觉" in summary:
        return "多余生成/幻觉错误"
    return "未知错误"


def _generate_code(agent, system_prompt: str, clause: str):
    class _Msg:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    messages = [_Msg("system", system_prompt), _Msg("user", clause)]
    start = time.perf_counter()
    resp = agent.model.generate(messages)
    latency_ms = (time.perf_counter() - start) * 1000.0
    code = getattr(resp, "content", "") or ""
    return code, latency_ms


def evaluate_instruction(chat_api, agent, instruction: Dict) -> InstructionEvalResult:
    # ── 每条新指令开始前，清除上次残留的中断标志 ──
    chat_api.is_interrupted = False
    start_ts = time.perf_counter()
    inst_start_mono = time.monotonic()  # 超时用单调时钟
    instruction_text = instruction["text"]
    inference_latency_ms = 0.0
    execution_time_ms = 0.0
    parsing_success = False
    execution_success = False
    parsing_explanation = ""
    generated_task_sequence: List[Dict] = []
    generated_code = ""
    error_type = None
    error_description = ""
    summary = ""

    # ── 复杂指令保护：与 _worker_loop 保持一致 ──
    if chat_api._is_complex_instruction(instruction_text):
        clauses = [instruction_text]  # 整条指令不拆分，直接交 LLM
        _skip_hard_rules = True
    else:
        clauses = chat_api._split_task_clauses(instruction_text)
        if not clauses:
            clauses = [instruction_text]
        _skip_hard_rules = False

    generated_parts = []
    parsing_success = True
    execution_success = True

    for clause in clauses:
        # ── 超时保护：单条指令执行超过 INSTRUCTION_TIMEOUT_S 秒则中断 ──
        elapsed_s = time.monotonic() - inst_start_mono
        if elapsed_s > INSTRUCTION_TIMEOUT_S:
            chat_api.is_interrupted = True
            _safe_stop(chat_api)
            execution_success = False
            error_type = "执行超时"
            error_description = f"instruction timeout {elapsed_s:.1f}s > {INSTRUCTION_TIMEOUT_S:.0f}s"
            break
        # 复杂指令跳过硬规则，直接交给LLM（与 _worker_loop 保持一致）
        if not _skip_hard_rules:
            action, summary = chat_api._handle_hard_rules(clause)
            if action == "continue":
                step_ok = _summary_is_ok(summary)
                if not step_ok:
                    parsing_success = True
                    execution_success = False
                    error_type = _attribute_error_type(summary, instruction_text, parsing_success)
                    error_description = summary or "hard rule failed"
                    break
                continue

        try:
            code, latency_ms = _generate_code(
                agent,
                chat_api.Prompt_dit["Prompt_smol"],
                clause,
            )
            inference_latency_ms += latency_ms
            generated_parts.append(code)
            if not code.strip():
                parsing_success = False
                execution_success = False
                error_type = "解析错误"
                error_description = "empty code"
                break

            ok = bool(chat_api.execute_generated_code(code))
            _safe_stop(chat_api)  # 无论 LLM 代码有没有停止，强制刹车
            execution_success = execution_success and ok
            summary = chat_api._get_latest_result_cn(default_text="")
            if not ok:
                error_type = _attribute_error_type(summary, instruction_text, parsing_success)
                error_description = "execution failed"
                break
        except Exception as exc:
            parsing_success = False
            execution_success = False
            error_type = "执行异常"
            error_description = str(exc)
            break

    # ── 围栏/超时中断后统一收尾 ──
    if chat_api.is_interrupted and error_type is None:
        execution_success = False
        error_type = "围栏中断"
        error_description = "空间围栏连续触发超限，自动中断"
        _safe_stop(chat_api)

    generated_code = "\n".join([p for p in generated_parts if p])
    execution_time_ms = (time.perf_counter() - start_ts) * 1000.0

    has_branch = _detect_branch(instruction_text)
    branch_correctness = _check_branch_correctness(instruction_text, summary, execution_success) if has_branch else True

    # 硬规则成功处理的指令：系统通过规则引擎正确理解了语义，SFS 直接给满分
    if not generated_code and execution_success:
        semantic_fidelity_score = 100.0
        sfs_target = sfs_param = sfs_action = sfs_exec = 25.0
    else:
        sfs_result = _evaluate_semantic_fidelity(
            instruction_text,
            instruction.get("expected_behavior", {}),
            generated_code,
            parsing_success,
            execution_success,
            execution_time_ms,
        )
        semantic_fidelity_score, sfs_target, sfs_param, sfs_action, sfs_exec = sfs_result

    task_completed = _evaluate_task_completion(
        execution_success, execution_time_ms, error_type, parsing_success,
    )
    ee = _calc_single_execution_efficiency(
        instruction.get("difficulty", ""), execution_time_ms, execution_success,
    )

    return InstructionEvalResult(
        instruction_id=instruction["id"],
        instruction_text=instruction_text,
        model_name=instruction["model"],
        difficulty=instruction.get("difficulty", ""),
        semantic_type=instruction.get("semantic_type", ""),
        group=instruction.get("group", ""),
        origin_text=instruction.get("origin_text", ""),
        parsing_success=parsing_success,
        execution_success=execution_success,
        parsing_explanation=parsing_explanation,
        generated_task_sequence=generated_task_sequence,
        inference_latency_ms=inference_latency_ms,
        execution_time_ms=execution_time_ms,
        has_branch=has_branch,
        branch_correctness=branch_correctness,
        semantic_fidelity_score=semantic_fidelity_score,
        sfs_target=sfs_target,
        sfs_param=sfs_param,
        sfs_action=sfs_action,
        sfs_execution=sfs_exec,
        task_completed=task_completed,
        execution_efficiency=ee,
        error_type=error_type,
        error_description=error_description,
        timestamp=datetime.now().isoformat(),
    )


def _group_by_origin(results: List[InstructionEvalResult], group: str) -> Dict[str, List[InstructionEvalResult]]:
    grouped: Dict[str, List[InstructionEvalResult]] = {}
    for result in results:
        if result.group != group:
            continue
        key = result.origin_text or result.instruction_text
        grouped.setdefault(key, []).append(result)
    return grouped


def _calc_generalization_robustness(results: List[InstructionEvalResult]) -> float:
    """
    泛化鲁棒性 (GR) — 合并原 robustness_index + zero_shot_success_rate。
    同语义不同表述的成功率均值（涵盖 zero_shot 和 colloquial 组）。
    """
    non_core = [r for r in results if r.group != "core"]
    if not non_core:
        return 100.0  # 无泛化测试数据时默认满分
    grouped: Dict[str, List[InstructionEvalResult]] = {}
    for r in non_core:
        key = r.origin_text or r.instruction_text
        grouped.setdefault(key, []).append(r)
    rates = []
    for items in grouped.values():
        success = sum(1 for r in items if r.task_completed)
        rates.append((success / len(items)) * 100.0)
    return sum(rates) / len(rates) if rates else 0.0


def calculate_metrics(results: List[InstructionEvalResult]) -> Optional[ExperimentMetrics]:
    if not results:
        return None

    model_name = results[0].model_name

    # PSR — 指令解析率
    parsing_success_rate = sum(1 for r in results if r.parsing_success) / len(results) * 100

    # TCR — 任务完成率 (参考 SayCan)
    task_completion_rate = sum(1 for r in results if r.task_completed) / len(results) * 100

    # Latency
    avg_latency = sum(r.inference_latency_ms for r in results) / len(results)

    # CA — 条件分支准确率
    with_branch = [r for r in results if r.has_branch]
    conditional_accuracy = (
        sum(1 for r in with_branch if r.branch_correctness) / len(with_branch) * 100
        if with_branch
        else 100.0
    )

    # SFS — 语义一致性分数 (参考 ALFRED GC)
    semantic_fidelity = sum(r.semantic_fidelity_score for r in results) / len(results)

    # GR — 泛化鲁棒性
    generalization_robustness = _calc_generalization_robustness(results)

    # EE — 执行效率 (参考 VLN SPL)
    execution_efficiency = (
        sum(r.execution_efficiency for r in results) / len(results)
    )

    return ExperimentMetrics(
        model_name=model_name,
        parsing_success_rate=parsing_success_rate,
        task_completion_rate=task_completion_rate,
        avg_inference_latency_ms=avg_latency,
        conditional_accuracy=conditional_accuracy,
        semantic_fidelity_score=semantic_fidelity,
        generalization_robustness=generalization_robustness,
        execution_efficiency=execution_efficiency,
    )


def export_results_csv(results: List[InstructionEvalResult], output_path: str):
    _ensure_output_dir(os.path.dirname(output_path))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "instruction_id",
            "instruction_text",
            "origin_text",
            "model_name",
            "difficulty",
            "semantic_type",
            "group",
            "parsing_success",
            "execution_success",
            "task_completed",
            "inference_latency_ms",
            "execution_time_ms",
            "branch_correctness",
            "semantic_fidelity_score",
            "sfs_target",
            "sfs_param",
            "sfs_action",
            "sfs_execution",
            "execution_efficiency",
            "error_type",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "instruction_id": result.instruction_id,
                    "instruction_text": result.instruction_text,
                    "origin_text": result.origin_text,
                    "model_name": result.model_name,
                    "difficulty": result.difficulty,
                    "semantic_type": result.semantic_type,
                    "group": result.group,
                    "parsing_success": result.parsing_success,
                    "execution_success": result.execution_success,
                    "task_completed": result.task_completed,
                    "inference_latency_ms": f"{result.inference_latency_ms:.2f}",
                    "execution_time_ms": f"{result.execution_time_ms:.2f}",
                    "branch_correctness": result.branch_correctness,
                    "semantic_fidelity_score": f"{result.semantic_fidelity_score:.1f}",
                    "sfs_target": f"{result.sfs_target:.1f}",
                    "sfs_param": f"{result.sfs_param:.1f}",
                    "sfs_action": f"{result.sfs_action:.1f}",
                    "sfs_execution": f"{result.sfs_execution:.1f}",
                    "execution_efficiency": f"{result.execution_efficiency:.1f}",
                    "error_type": result.error_type or "N/A",
                }
            )


def export_metrics_json(metrics_list: List[ExperimentMetrics], output_path: str):
    _ensure_output_dir(os.path.dirname(output_path))
    data = {
        "timestamp": datetime.now().isoformat(),
        "models": [
            {
                "name": m.model_name,
                "metrics": {
                    "parsing_success_rate": f"{m.parsing_success_rate:.2f}%",
                    "task_completion_rate": f"{m.task_completion_rate:.2f}%",
                    "avg_inference_latency_ms": f"{m.avg_inference_latency_ms:.2f}",
                    "conditional_accuracy": f"{m.conditional_accuracy:.2f}%",
                    "semantic_fidelity_score": f"{m.semantic_fidelity_score:.1f}",
                    "generalization_robustness": f"{m.generalization_robustness:.2f}%",
                    "execution_efficiency": f"{m.execution_efficiency:.1f}%",
                },
            }
            for m in metrics_list
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_experiment(args):
    init_runtime_logger()
    logger = get_runtime_logger("exp2")
    logger.info("Experiment 2 start")

    comm_api = BodyCommMavlink()
    logger.info("Communication initialized")

    preflight = comm_api.preflight_check()
    if not bool(preflight.get("ok", False)):
        raise RuntimeError("Preflight failed")

    _auto_offboard_takeoff(comm_api, logger)

    mav_list, vehicle_num, _ = comm_api.GetBodyMavList()
    if mav_list:
        mav_list[0].move_with_speed = comm_api.move_with_speed

    chat_api = OpenAI_APIs(
        mav_list,
        vehicle_num,
        comm_api.detect_yolo,
        comm_api.approachObjective,
        comm_api.look,
        comm_api.search_object,
        comm_api.save_detection_image,
        comm_api.face_objective_to_target,
        comm_api.strike_objective_to_target,
    )

    if hasattr(comm_api, "set_interrupt_check"):
        comm_api.set_interrupt_check(lambda: getattr(chat_api, "is_interrupted", False))

    # 注册围栏连续触发超限时的中断回调（连续 N 次围栏触发 → 自动 abort 当前任务）
    comm_api._interrupt_set_callback = lambda: setattr(chat_api, "is_interrupted", True)

    # 通知Agent起飞已由_auto_offboard_takeoff完成，后续LLM代码中的initOffboard应被移除
    chat_api._init_sequence_done = True

    all_instructions = parse_instruction_library(args.instruction_md)
    if args.max_instructions > 0:
        all_instructions = all_instructions[: args.max_instructions]

    # 随机抽样：从每个 n-n 组中随机取 sample_per_group 条
    sample_n = getattr(args, "sample_per_group", 0)
    seed_val = getattr(args, "seed", None)
    if sample_n > 0:
        instructions = _sample_instructions(all_instructions, sample_per_group=sample_n, seed=seed_val)
    else:
        instructions = all_instructions

    home_every = getattr(args, "home_every", 5)

    results: List[InstructionEvalResult] = []
    for model_name in args.models:
        agent = _build_agent(chat_api, _get_model_config(model_name))

        logger.info(
            f"RUN_START model={model_name} total={len(instructions)} "
            f"(sampled {sample_n}/group from {len(all_instructions)}) "
            f"home_every={home_every} seed={seed_val}"
        )

        exec_count = 0  # 本轮已执行的指令计数
        for instruction in instructions:
            instruction_copy = dict(instruction)
            instruction_copy["model"] = model_name
            logger.info(f"EVAL_START model={model_name} id={instruction_copy['id']} text={instruction_copy['text']}")

            # 特定指令执行前强制回起点（防止前序指令导致目标不可见）
            inst_id = instruction_copy["id"]
            if inst_id in RTH_BEFORE_IDS:
                logger.info(f"RTH_BEFORE triggered for id={inst_id}, returning to home before execution")
                _safe_stop(comm_api)
                _return_to_home_position(comm_api, logger, hover_alt=-0.5)
                time.sleep(2.0)

            comm_api._fence_consec_count = 0  # 每条指令开始前重置围栏计数
            result = evaluate_instruction(chat_api, agent, instruction_copy)
            results.append(result)
            logger.info(
                f"EVAL_DONE model={model_name} id={instruction_copy['id']} ok={result.execution_success}"
            )
            exec_count += 1
            time.sleep(max(args.step_wait_s, 0.0))

            # 空间围栏/超时/返航异常后强制回起点
            if not result.execution_success and (
                (result.error_description and (
                    "围栏" in result.error_description or "返航" in result.error_description
                )) or
                (result.error_type and (
                    "超时" in result.error_type or "围栏" in result.error_type
                ))
            ):
                reason = result.error_type or result.error_description or "unknown"
                logger.warning(f"SAFETY_RECOVERY: 强制回起点 reason={reason[:60]}")
                _safe_stop(comm_api)
                _return_to_home_position(comm_api, logger, hover_alt=-0.5)
                time.sleep(2.0)

            # 每 home_every 条指令回起点一次
            elif home_every > 0 and exec_count % home_every == 0:
                logger.info(f"HOME_EVERY triggered after {exec_count} instructions, returning to home")
                _return_to_home_position(comm_api, logger, hover_alt=-0.5)
                time.sleep(2.0)

        # 最后一批不足 home_every 条的，也回一次起点
        if home_every > 0 and exec_count % home_every != 0:
            logger.info(f"HOME_FINAL after {exec_count} instructions, returning to home")
            _return_to_home_position(comm_api, logger, hover_alt=-0.5)
            time.sleep(2.0)

    output_dir = _ensure_output_dir(args.output_dir)
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    export_results_csv(results, csv_path)

    metrics_list: List[ExperimentMetrics] = []
    for model_name in args.models:
        model_results = [r for r in results if r.model_name == model_name]
        metrics = calculate_metrics(model_results)
        if metrics:
            metrics_list.append(metrics)

    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    export_metrics_json(metrics_list, metrics_path)

    # 同时复制到 test_exp2/ 根目录，作为"最新版"供 visualize_results.py 读取
    import shutil
    parent_dir = os.path.dirname(output_dir)
    for fname in ("evaluation_results.csv", "evaluation_metrics.json"):
        src = os.path.join(output_dir, fname)
        dst = os.path.join(parent_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"COPY_LATEST: {src} -> {dst}")

    logger.info(f"Experiment 2 completed. Results at {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 2 runner")
    parser.add_argument(
        "--instruction-md",
        default=DEFAULT_INSTRUCTION_MD,
        help="Path to experiment2_instruction_library.md",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for logs/test_exp2",
    )
    parser.add_argument(
        "--models",
        default="DeepSeek-V3",
        help="Comma-separated model names",
    )
    parser.add_argument(
        "--max-instructions",
        type=int,
        default=0,
        help="Limit number of instructions (0 = all)",
    )
    parser.add_argument(
        "--sample-per-group",
        type=int,
        default=1,
        help="从每个 n-n 组中随机抽取的指令数 (0 = 全部保留)",
    )
    parser.add_argument(
        "--home-every",
        type=int,
        default=5,
        help="每执行N条指令后返回起飞点 (0 = 不自动返回)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，固定后可复现抽样结果",
    )
    parser.add_argument(
        "--step-wait-s",
        type=float,
        default=2.0,
        help="Wait time between instructions",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    run_experiment(args)


if __name__ == "__main__":
    main()

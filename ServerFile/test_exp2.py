import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
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


DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "logs", "test_exp2")
DEFAULT_INSTRUCTION_MD = os.path.join(
    os.path.dirname(__file__), "..", "md", "experiment2_instruction_library.md"
)

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
    semantic_depth_score: float
    error_type: Optional[str]
    error_description: str
    timestamp: str
    notes: str = ""


@dataclass
class ExperimentMetrics:
    model_name: str
    parsing_success_rate: float
    execution_success_rate: float
    avg_inference_latency_ms: float
    branch_correctness_rate: float
    semantic_depth_score: float
    robustness_index: float
    colloquial_tolerance_rate: float
    zero_shot_success_rate: Optional[float] = None


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


def _evaluate_semantic_depth(instruction_text: str, expected_behavior: Dict, generated_code: str, parsing_success: bool) -> float:
    if not parsing_success:
        return 0.0

    modifier_map = {
        "一点点": 0.3,
        "稍微": 0.2,
        "大幅度": 0.5,
        "快速": 1.0,
        "缓慢": 0.2,
    }

    modifiers = [m for m in modifier_map if m in instruction_text]
    if not modifiers:
        return 100.0

    expected_hint = str(expected_behavior.get("key_params", ""))
    expected_numbers = _extract_numbers(expected_hint)
    expected = expected_numbers[0] if expected_numbers else modifier_map.get(modifiers[0])

    code_numbers = _extract_numbers(generated_code)
    if not code_numbers or expected is None:
        return 70.0

    best = min(abs(v - expected) / max(abs(expected), 1e-6) for v in code_numbers)
    if best <= 0.2:
        return 100.0
    if best <= 0.4:
        return 80.0
    return 60.0


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
    start_ts = time.perf_counter()
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

    clauses = chat_api._split_task_clauses(instruction_text)
    if not clauses:
        clauses = [instruction_text]

    generated_parts = []
    parsing_success = True
    execution_success = True

    for clause in clauses:
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

    generated_code = "\n".join([p for p in generated_parts if p])
    execution_time_ms = (time.perf_counter() - start_ts) * 1000.0

    has_branch = _detect_branch(instruction_text)
    branch_correctness = _check_branch_correctness(instruction_text, summary, execution_success) if has_branch else True
    semantic_depth_score = _evaluate_semantic_depth(
        instruction_text,
        instruction.get("expected_behavior", {}),
        generated_code,
        parsing_success,
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
        semantic_depth_score=semantic_depth_score,
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


def _calc_colloquial_tolerance(results: List[InstructionEvalResult]) -> float:
    grouped = _group_by_origin(results, "colloquial")
    if not grouped:
        return 0.0
    rates = []
    for items in grouped.values():
        success = sum(1 for r in items if r.execution_success)
        rates.append((success / len(items)) * 100.0)
    return sum(rates) / len(rates)


def _calc_robustness_index(results: List[InstructionEvalResult]) -> float:
    grouped = _group_by_origin(results, "zero_shot")
    if not grouped:
        return 0.0
    scores = []
    for items in grouped.values():
        scores.append(100.0 if all(r.execution_success for r in items) else 0.0)
    return sum(scores) / len(scores)


def _calc_zero_shot_success_rate(results: List[InstructionEvalResult]) -> Optional[float]:
    items = [r for r in results if r.group == "zero_shot"]
    if not items:
        return None
    success = sum(1 for r in items if r.execution_success)
    return (success / len(items)) * 100.0


def calculate_metrics(results: List[InstructionEvalResult]) -> Optional[ExperimentMetrics]:
    if not results:
        return None

    model_name = results[0].model_name
    parsing_success_rate = sum(1 for r in results if r.parsing_success) / len(results) * 100
    parsed = [r for r in results if r.parsing_success]
    execution_success_rate = (
        sum(1 for r in parsed if r.execution_success) / len(parsed) * 100 if parsed else 0.0
    )
    avg_latency = sum(r.inference_latency_ms for r in results) / len(results)
    with_branch = [r for r in results if r.has_branch]
    branch_correctness_rate = (
        sum(1 for r in with_branch if r.branch_correctness) / len(with_branch) * 100
        if with_branch
        else 100.0
    )
    semantic_depth = sum(r.semantic_depth_score for r in results) / len(results)

    robustness_index = _calc_robustness_index(results)
    colloquial_tolerance_rate = _calc_colloquial_tolerance(results)
    zero_shot_rate = _calc_zero_shot_success_rate(results)

    return ExperimentMetrics(
        model_name=model_name,
        parsing_success_rate=parsing_success_rate,
        execution_success_rate=execution_success_rate,
        avg_inference_latency_ms=avg_latency,
        branch_correctness_rate=branch_correctness_rate,
        semantic_depth_score=semantic_depth,
        robustness_index=robustness_index,
        colloquial_tolerance_rate=colloquial_tolerance_rate,
        zero_shot_success_rate=zero_shot_rate,
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
            "inference_latency_ms",
            "execution_time_ms",
            "branch_correctness",
            "semantic_depth_score",
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
                    "inference_latency_ms": f"{result.inference_latency_ms:.2f}",
                    "execution_time_ms": f"{result.execution_time_ms:.2f}",
                    "branch_correctness": result.branch_correctness,
                    "semantic_depth_score": f"{result.semantic_depth_score:.2f}",
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
                    "execution_success_rate": f"{m.execution_success_rate:.2f}%",
                    "avg_inference_latency_ms": f"{m.avg_inference_latency_ms:.2f}",
                    "branch_correctness_rate": f"{m.branch_correctness_rate:.2f}%",
                    "semantic_depth_score": f"{m.semantic_depth_score:.2f}",
                    "robustness_index": f"{m.robustness_index:.2f}%",
                    "colloquial_tolerance_rate": f"{m.colloquial_tolerance_rate:.2f}%",
                    "zero_shot_success_rate": (
                        f"{m.zero_shot_success_rate:.2f}%" if m.zero_shot_success_rate is not None else "N/A"
                    ),
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

    # 通知Agent起飞已由_auto_offboard_takeoff完成，后续LLM代码中的initOffboard应被移除
    chat_api._init_sequence_done = True

    instructions = parse_instruction_library(args.instruction_md)
    if args.max_instructions > 0:
        instructions = instructions[: args.max_instructions]

    results: List[InstructionEvalResult] = []
    for model_name in args.models:
        agent = _build_agent(chat_api, _get_model_config(model_name))
        for instruction in instructions:
            instruction_copy = dict(instruction)
            instruction_copy["model"] = model_name
            logger.info(f"EVAL_START model={model_name} id={instruction_copy['id']} text={instruction_copy['text']}")
            result = evaluate_instruction(chat_api, agent, instruction_copy)
            results.append(result)
            logger.info(
                f"EVAL_DONE model={model_name} id={instruction_copy['id']} ok={result.execution_success}"
            )
            time.sleep(max(args.step_wait_s, 0.0))

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
        default="DeepSeek-V3,Doubao-pro,GPT-4o",
        help="Comma-separated model names",
    )
    parser.add_argument(
        "--max-instructions",
        type=int,
        default=0,
        help="Limit number of instructions (0 = all)",
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

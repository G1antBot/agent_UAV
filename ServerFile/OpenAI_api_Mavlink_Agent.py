# ChatGpt交互模式控制类
'''
注意：本代码采用无人机的NED坐标系，室内动捕系统环境下飞行时，定义N向为动捕系统的X轴正方向，地面为高度0，向上为负
'''

import os
import ast
import time
import json
import openai
import numpy as np
import cv2
import re
import traceback
import math
import sys
import types
import threading
import queue
import pathlib

from datetime import datetime, timezone
from Description import Description as Des
from Coordinate_Transformation import body_to_ned as b2n
from runtime_logger import get_runtime_logger
from smolagents import CodeAgent, PromptTemplates, PlanningPromptTemplate, ManagedAgentPromptTemplate, \
    FinalAnswerPromptTemplate
from volcEngineLLM import VolcEngineFakeHFModel


class OpenAI_APIs(Des):
    version = "3.2"

    def __init__(self, MavList, VehilceNum, detect_function, approachObjective_function, look_function,
                 search_object_function, save_detection_image_function=None, face_objective_function=None,
                 strike_objective_function=None):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化无人机列表和数量
        self.MavList = MavList
        self.VehilceNum = VehilceNum
        # 初始化功能函数，用于无人机的特定操作
        self.detect_function = detect_function
        self.approachObjective_function = approachObjective_function
        self.look_function = look_function
        self.search_object_function = search_object_function
        self.save_detection_image_function = save_detection_image_function
        self.face_objective_function = face_objective_function
        self.strike_objective_function = strike_objective_function

        # 设置火山引擎API密钥
        os.environ['OPENAI_API_KEY'] = "24572520-5c64-4470-8c3d-5ecb84781725"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # 设置火山引擎API的基础URL
        self.client = openai.OpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3 ")
        # 设置使用的语言模型
        self.LLMModel = "deepseek-v3-250324"
        # 初始化聊天历史记录
        self.chatHistory = []
        self.logger = get_runtime_logger("agent")

        # 读取 Config.json 获取硬规则开关
        self.enable_hard_rule_routing = True
        self.semantic_distance_map = {
            "大幅度": 0.5,
            "多一点": 0.5,
            "靠近一点点": 0.3,
            "一点点": 0.3,
            "稍微挪一点": 0.2,
            "稍微": 0.2,
        }
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'Config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    content = re.sub(r'//.*', '', content)
                    config_data = json.loads(content)
                    self.enable_hard_rule_routing = config_data.get("agent_config", {}).get("enable_hard_rule_routing", True)
                    distance_map = config_data.get("agent_config", {}).get("semantic_distance_map", {})
                    if isinstance(distance_map, dict) and distance_map:
                        self.semantic_distance_map = distance_map
        except Exception as e:
            self.logger.warning(f"读取 Config.json 失败: {e}")

        # 启动简洁输出窗口（可选，失败时静默降级）
        try:
            from simple_output_window import OutputWindow
            try:
                self.output_window = OutputWindow()
            except Exception as e:
                self.output_window = None
                self.logger.info(f"创建简洁输出窗口失败: {e}")
        except Exception:
            self.output_window = None

        # ── 双线程架构：看门狗 + Agent工人 ──────────────────────────────
        # 任务队列：常规指令在此排队，顺序交给 Agent Worker 执行
        self._task_queue = queue.Queue()
        # 全局中断标志：急停时置 True，所有耗时函数应检查并中止
        self.is_interrupted = False
        # 程序终止标志：Main_Control 退出时置 True，两个线程均会退出
        self._stop_signal = False
        # ────────────────────────────────────────────────────────────────

        self.logger.info(f"OpenAI_APIs 初始化完成, enable_hard_rule_routing={self.enable_hard_rule_routing}")
        # 当前任务 id（用于把同一次任务的生成代码写入同一文件）
        self._current_cmd_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 最近一次生成代码文件路径
        self._last_generated_code_path = None
        # 是否已执行过场外模式启动序列（initOffboard + SendPosNED）
        self._init_sequence_done = False

    def _get_latest_result_cn(self, default_text: str = "执行完成"):
        """
        从通信模块读取最近一次任务结果中文描述；若不可用则返回默认值。
        """
        try:
            comm_obj = getattr(self.search_object_function, "__self__", None)
            if comm_obj is None:
                return default_text
            result_cn = getattr(comm_obj, "last_search_result_cn", "")
            return result_cn if result_cn else default_text
        except Exception:
            return default_text

    def _emit_highlight_block(self, title: str, lines, ok: bool = True):
        """
        在终端与日志同时输出重点结果块（前缀 + 方框）。
        """
        safe_lines = [str(x) for x in (lines or [])]
        edge = "=" * 17
        header = f"{edge} {title} {edge}"
        footer = "=" * len(header)

        log_fn = self.logger.info if ok else self.logger.warning
        log_fn(f">>> {title}")
        log_fn(header)
        for ln in safe_lines:
            log_fn(ln)
        log_fn(footer)

    def _emit_step_result(self, cmd_id: str, idx: int, total: int, mode_cn: str, success: bool, summary: str):
        status_cn = "成功" if success else "失败"
        lines = [
            f"任务编号: {cmd_id}",
            f"步骤: {idx}/{total}",
            f"执行方式: {mode_cn}",
            f"状态: {status_cn}",
            f"结果: {summary}",
        ]
        self._emit_highlight_block("步骤结果", lines, ok=success)

    def _reset_comm_task_timeout(self, clause: str = ""):
        """在每个子句执行前重置通信层任务超时基准。"""
        try:
            comm_obj = getattr(self.search_object_function, "__self__", None)
            if comm_obj is None:
                return
            reset_fn = getattr(comm_obj, "set_task_start_timestamp", None)
            if callable(reset_fn):
                reset_fn()
                if clause:
                    self.logger.info(f"重置通信层任务计时: clause={clause}")
        except Exception as e:
            self.logger.warning(f"重置通信层任务计时失败: {e}")

    def _handle_hard_rules(self, task: str):
        """
        全量硬规则层：处理退出/急停与各类基础动作模板。
        如果 enable_hard_rule_routing 为 False，则仅保留退出和急停，其余全部交给LLM。
        返回(action, summary)
        action取值："continue" 表示已处理并进入下一轮；"pass" 表示交给LLM。
        """
        text = (task or "").strip()
        if not text:
            return "continue", "空指令"

        # 最高安全级别：无论开关是否打开，退出和急停始终生效
        if text.lower() in self.ExitList:
            print("对话结束，程序退出。")
            self.logger.info("用户主动退出")
            raise KeyboardInterrupt

        # 紧急停止词库（看门狗线程也使用此正则）
        if re.search(
            r"急停|紧急停止|立即停止|快停|停下|停止|悬停|stop\b|halt\b",
            text, flags=re.IGNORECASE
        ):
            self._emergency_stop()
            return "continue", "紧急停止已执行"

        # 条件模板（优先级高于常规硬规则）
        handled, ok, summary = self._handle_conditional_task(text)
        if handled:
            return "continue", summary

        # 状态检查：如果硬规则路由关闭，则剩余动作交给LLM
        if not getattr(self, 'enable_hard_rule_routing', True):
            return "pass", "交由LLM处理"

        # === 以下为受开关控制的本地硬规则路由 ===

        # 1. 基础位移硬规则
        move_parsed = self._parse_body_move_clause(text)
        if move_parsed is not None:
            if move_parsed.get("type") == "move_invalid":
                reason = move_parsed.get("reason", "位移方向冲突")
                print(f"执行失败：{reason}")
                self.logger.warning(f"基础位移硬规则拒绝执行: {reason}")
                return "continue", reason
            ok = self._execute_body_move_template(
                dx_body=move_parsed["dx_body"],
                dy_body=move_parsed["dy_body"],
                dz_body=move_parsed["dz_body"],
                distance_m=move_parsed["distance_m"],
                direction_text=move_parsed["direction_text"],
            )
            if ok:
                return "continue", "基础位移执行完成"
            return "continue", "基础位移执行失败"

        # 2. 转向硬规则
        turn_parsed = self._parse_turn_clause(text)
        if turn_parsed is not None:
            ok = self._execute_turn_template(
                sign=turn_parsed["sign"],
                deg=turn_parsed["deg"],
                turn_text=turn_parsed["turn_text"],
            )
            if ok:
                return "continue", "转向执行完成"
            return "continue", "转向执行失败"

        # 3. 搜索硬规则
        search_parsed = self._parse_search_clause(text)
        if search_parsed is not None:
            ok = self._execute_search_template(
                object_name=search_parsed["object_name"],
                mode=search_parsed["mode"],
            )
            if ok:
                return "continue", self._get_latest_result_cn(default_text="搜索完成")
            return "continue", self._get_latest_result_cn(default_text="搜索失败")

        # 4. 靠近硬规则
        approach_parsed = self._parse_approach_clause(text)
        if approach_parsed is not None:
            if "distance_m" in approach_parsed:
                ok = self._execute_approach_distance_template(
                    approach_parsed["object_name"],
                    approach_parsed["distance_m"],
                )
            else:
                ok = self._execute_approach_template(approach_parsed["object_name"], spatial_hint=text)
            if ok:
                return "continue", self._get_latest_result_cn(default_text="靠近完成")
            return "continue", self._get_latest_result_cn(default_text="靠近失败")

        # 5. 原地朝向硬规则
        face_parsed = self._parse_face_object_clause(text)
        if face_parsed is not None:
            ok = self._execute_face_object_template(face_parsed["object_name"])
            if ok:
                return "continue", "朝向执行完成"
            return "continue", "朝向执行失败"

        # 6. 回到起飞点硬规则
        if self._parse_return_home_clause(text):
            ok = self._execute_return_home_template()
            if ok:
                return "continue", "已返回起飞点"
            return "continue", "返回起飞点失败"

        # 都未命中则交由 LLM 处理
        return "pass", "交由LLM处理"

    def _guard_check_deadline(self):
        """任务看门狗：超时后立即急停并中止本轮执行。"""
        state = getattr(self, "_task_guard_state", None)
        if not state:
            return
        deadline = state.get("deadline")
        if deadline is None:
            return
        if time.monotonic() > deadline:
            try:
                self.MavList[0].SendVelFRD(0, 0, 0, 0)
            except Exception:
                pass
            raise RuntimeError("任务执行超时，已触发急停保护")

    @staticmethod
    def _validate_target_name(target):
        """仅允许非空字符串目标名，拒绝纯数字/坐标类输入。"""
        if not isinstance(target, str):
            return None
        text = target.strip()
        if not text:
            return None
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
            return None
        if re.fullmatch(r"\[.*\]|\(.*\)", text):
            return None
        return text

    @staticmethod
    def _normalize_object_alias(name):
        """归一化常见目标别名，减少LLM在英文/中文目标名上的漂移。
        两级匹配：
        1. 精确匹配：直接查表
        2. 模糊包含匹配：当LLM传入"最近的红色气球"等带修饰词的目标名时，
           提取核心类别关键词并映射到YOLOE能识别的标准名
        """
        if not isinstance(name, str):
            return name
        text = name.strip()

        # === 第一级：精确匹配（覆盖常见写法和LLM常见漂移） ===
        alias_map = {
            "drone": "uav", "无人机": "uav", "uav": "uav",
            "balloon": "balloon", "气球": "balloon",
            "red balloon": "red balloon", "red_balloon": "red balloon",
            "blue ball": "blue ball", "blue_ball": "blue ball",
            "小球": "blue ball", "蓝色小球": "blue ball",
            "红色气球": "red balloon", "红气球": "red balloon",
            "car": "car", "小车": "car", "汽车": "car", "车辆": "car",
        }
        exact = alias_map.get(text.lower(), alias_map.get(text, None))
        if exact is not None:
            return exact

        # === 第二级：模糊包含匹配 ===
        # 按从长到短排序，避免"气球"先于"红色气球"匹配
        fuzzy_rules = [
            # (关键词列表, 映射目标)  —— 中英文混合，长词优先
            (["红色气球", "红气球", "red balloon", "red_balloon"], "red balloon"),
            (["蓝色小球", "蓝色球", "蓝球", "blue ball", "blue_ball"], "blue ball"),
            (["气球", "balloon"],  "balloon"),
            (["小球", "ball"],     "blue ball"),
            (["小车", "车辆", "汽车", "car"], "car"),
            (["无人机", "飞机", "uav", "drone"], "uav"),
        ]
        lower = text.lower()
        for keywords, target in fuzzy_rules:
            for kw in keywords:
                if kw in text or kw in lower:
                    return target

        # 都没匹配到，原样返回
        return text

    def _split_task_clauses(self, task: str):
        """
        将一条指令切分为多个子句，便于对子句进行模板拦截。
        """
        if not task:
            return []
        # 先按连接词切，再按常见标点切
        clauses = re.split(r"(?:然后|再|并且|并|,|，|;|；|。)", task)
        return [c.strip() for c in clauses if c and c.strip()]

    def _is_complex_instruction(self, task: str) -> bool:
        """检测指令是否包含语义耦合信号，决定是否跳过子句拆分。
        命中任一信号即返回 True，整条指令将不被拆分而直接交给 LLM。
        """
        text = (task or "").strip()
        if not text:
            return False

        # 1. 条件/分支信号
        if re.search(r"如果|若是|若(?!干)|有的话|没有就|否则|根据|找到.*的话|找不到.*的话", text):
            return True

        # 2. 约束耦合信号（动作 + 约束在同一指令中）
        if re.search(r"同时|过程中|前提下|情况下|期间", text):
            return True

        # 3. 自适应/动态调整信号
        if re.search(r"自适应|自动调|远.*快.*近.*慢", text):
            return True

        # 4. 循环终止条件
        if re.search(r"直到|一边.*一边|一旦", text):
            return True

        # 5. 动态速度修饰（动作 + 速度变化描述不可拆分）
        if re.search(r"(?:减速|降速|加速|递减|递增|逐步|逐渐).*(?:靠近|接近|飞向|前进|后退|左转|右转)", text):
            return True
        if re.search(r"(?:靠近|接近|飞向|前进|后退|左转|右转).*(?:减速|降速|加速|递减|递增|逐步|逐渐|速度)", text):
            return True

        # 6. 连续搜索+靠近的复合语义
        if re.search(r"(?:搜索|找到|找).*(?:靠近|接近|飞向).*(?:靠近|接近|飞向|减速|加速|悬停)", text):
            return True

        # 7. 时序复合动作（先做A再做B，方向可能相反）
        if re.search(r"(?:前进|飞行|移动)\d+.*?(?:后|再|然后).*?(?:返回|原路|回来)", text):
            return True

        # 8. 渐变语义修饰（一点点/慢慢等修饰靠近/远离动作）
        if re.search(r"(?:一点点|慢慢|缓慢|小心).*?(?:靠近|接近|飞向|远离)", text):
            return True
        if re.search(r"(?:靠近|接近|飞向|远离).*?(?:一点点|慢慢|缓慢|小心)", text):
            return True

        return False

    def _parse_body_move_clause(self, clause: str):
        """
        解析前后左右位移子句。
        支持同义词与组合方向：
        - 前/后/左/右/上/下
        - 前进/后退/左移/右移/上升/下降
        允许小数，单位米，且组合方向按总位移长度固定为X处理。
        """
        if not clause:
            return None

        text = re.sub(r"\s+", "", clause)

        # 严格语义过滤：如果包含任何对速度或方式的额外修饰词，放弃硬规则拦截，抛给大模型
        if re.search(r"(速度|快|慢|以|用|每秒|m/s)", text):
            self.logger.info(f"硬规则路由放行: 发现复杂语义修饰 '{text}'")
            return None

        # 中文数字→阿拉伯数字归一化，支持"两米""三米""半米""一点五米"等口语表达
        _cn_digit_map = {
            "零": "0", "一": "1", "二": "2", "两": "2", "三": "3",
            "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
        }
        norm_text = text
        # "半米" → "0.5米"
        norm_text = re.sub(r"半米", "0.5米", norm_text)
        # "X点Y米" → "X.Y米"（如 "一点五米" → "1.5米"）
        def _cn_dot_repl(m):
            a = _cn_digit_map.get(m.group(1), m.group(1))
            b = _cn_digit_map.get(m.group(2), m.group(2))
            return f"{a}.{b}米"
        norm_text = re.sub(r"([零一二两三四五六七八九])点([零一二两三四五六七八九])米", _cn_dot_repl, norm_text)
        # 单个中文数字 + 米 → 阿拉伯数字 + 米
        for cn, ar in _cn_digit_map.items():
            norm_text = norm_text.replace(f"{cn}米", f"{ar}米")
        # "十米" → "10米"
        norm_text = re.sub(r"十米", "10米", norm_text)

        # 提取位移距离
        dist_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:米|m\b)", norm_text, re.IGNORECASE)
        if not dist_match:
            return None
        distance = float(dist_match.group(1))

        # 同义词归一化到六个方向字：前后左右上下
        normalize_pairs = [
            (r"向前飞|向前|前进|前移|往前", "前"),
            (r"向后飞|向后|后退|后移|往后", "后"),
            (r"向左飞|向左|左移|往左", "左"),
            (r"向右飞|向右|右移|往右", "右"),
            (r"向上飞|向上|上升|往上|升高", "上"),
            (r"向下飞|向下|下降|往下|降低", "下"),
        ]
        canon = text
        for p, repl in normalize_pairs:
            canon = re.sub(p, repl, canon)

        has_front = "前" in canon
        has_back = "后" in canon
        has_left = "左" in canon
        has_right = "右" in canon
        has_up = "上" in canon
        has_down = "下" in canon

        # 冲突方向直接拒绝
        if (has_front and has_back) or (has_left and has_right) or (has_up and has_down):
            return {"type": "move_invalid", "reason": "存在互斥方向"}

        # 机体系分量：x前、y右、z下（按当前坐标系约定）
        dx = (1.0 if has_front else 0.0) + (-1.0 if has_back else 0.0)
        dy = (1.0 if has_right else 0.0) + (-1.0 if has_left else 0.0)
        dz = (1.0 if has_down else 0.0) + (-1.0 if has_up else 0.0)

        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm == 0:
            return None

        # 组合方向按总位移长度固定为X
        unit_dx, unit_dy, unit_dz = dx / norm, dy / norm, dz / norm
        return {
            "type": "move",
            "distance_m": distance,
            "dx_body": unit_dx * distance,
            "dy_body": unit_dy * distance,
            "dz_body": unit_dz * distance,
            "direction_text": f"dx={unit_dx:.3f},dy={unit_dy:.3f},dz={unit_dz:.3f}",
        }

    def _parse_turn_clause(self, clause: str):
        """
        解析左转/右转子句，单位度。
        """
        if not clause:
            return None
        text = re.sub(r"\s+", "", clause)
        # 中文数字角度归一化: "九十度"→"90度", "四十五度"→"45度", "一百八十度"→"180度"
        _cn_d = {"零": "0", "一": "1", "二": "2", "两": "2", "三": "3",
                 "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
        def _cn_angle_repl(m):
            raw = m.group(1)
            # 常见整数角度快速映射
            _angle_map = {"九十": "90", "四十五": "45", "一百八十": "180", "一百八": "180",
                          "三十": "30", "六十": "60", "一百二十": "120"}
            if raw in _angle_map:
                return _angle_map[raw] + "度"
            # 单字中文数字
            if raw in _cn_d:
                return _cn_d[raw] + "度"
            return m.group(0)
        text = re.sub(r"([零一二两三四五六七八九十百]+)度", _cn_angle_repl, text)
        # 允许"向左转45度 / 向左转向45度 / 右转45度"等口语变体
        m = re.search(r"(左转|向左转|右转|向右转)[^0-9]{0,3}([0-9]+(?:\.[0-9]+)?)度", text)
        if not m:
            return None

        turn_text = m.group(1)
        deg = float(m.group(2))
        # 依据当前模拟器实际表现：左转应为负、右转应为正
        sign = -1.0 if turn_text in ("左转", "向左转") else 1.0
        return {"type": "turn", "deg": deg, "sign": sign, "turn_text": turn_text}

    def _parse_face_object_clause(self, clause: str):
        """
        解析“转向/朝向/面向某物体处”这类原地朝向目标的子句。
        """
        if not clause:
            return None

        text = re.sub(r"\s+", "", clause)
        m = re.search(r"(?:转向|朝向|面向)(.+?)?(?:处|方向|那里|那边)?$", text)
        if not m:
            return None

        object_name = (m.group(1) or "").strip()
        if not object_name:
            return None

        # 去掉常见空泛后缀，避免把“处/方向”带进目标名
        object_name = re.sub(r"(?:处|方向|那里|那边)$", "", object_name).strip()
        if not object_name:
            return None

        return {"type": "face_object", "object_name": object_name}

    def _parse_approach_clause(self, clause: str):
        """
        解析靠近子句：靠近/接近/飞向某物体。
        """
        if not clause:
            return None

        text = re.sub(r"\s+", "", clause)

        # ── 前置守卫：靠近/接近后面紧跟动作修饰（速度/时序词），说明是复合语义，交 LLM ──
        if re.search(r"(?:靠近|接近|飞向).*?(?:时|前|后|中|过程).*?(?:减速|降速|加速|慢慢|逐步|逐渐)", text):
            self.logger.info(f"硬规则路由放行: 发现复杂语义修饰 '{text}'")
            return None

        modifier_key, distance_m = self._match_distance_modifier(text)

        object_name = ""
        pre_match = re.search(r"(?:往|朝)(.+?)(?:方向)?(?:靠近|接近|飞向)", text)
        if pre_match:
            object_name = (pre_match.group(1) or "").strip()
        else:
            m = re.search(r"(?:靠近|接近|飞向)(.+?)$", text)
            if m:
                object_name = (m.group(1) or "").strip()
            else:
                # "去小车那里" / "到红色气球那边" 等口语化靠近表述
                m2 = re.search(r"(?:去|到)(.+?)(?:那里|那边|那儿|那去)$", text)
                if m2:
                    object_name = (m2.group(1) or "").strip()
                else:
                    return None

        if modifier_key:
            object_name = object_name.replace(modifier_key, "").strip()
        for key in sorted((self.semantic_distance_map or {}).keys(), key=len, reverse=True):
            if key:
                object_name = object_name.replace(key, "").strip()
        object_name = re.sub(r"方向$", "", object_name).strip()
        object_name = re.sub(r"^(?:一下|下|一个|目标|物体)", "", object_name).strip()
        object_name = re.sub(r"(?:目标|物体|并停下|再停下|后停下)$", "", object_name).strip()
        # 清除尾部距离修饰（如"到1米"、"到2m"）——这是靠近参数而非目标名
        object_name = re.sub(r"到\d+(?:\.\d+)?(?:米|m|厘米|cm)$", "", object_name).strip()
        if not object_name:
            return None

        # ── 后置守卫：拒绝非实体目标名（代词/动作碎片/修饰碎片） ──
        if re.match(r"^(?:它|它们|这个|那个|这里|那里)", object_name):
            self.logger.info(f"靠近硬规则放行: 目标以代词开头 '{object_name}'，交由LLM")
            return None
        # 如果提取出的"目标"包含动作动词/时序词，说明是误提取的碎片
        if re.search(r"减速|降速|加速|悬停|远离|搜索|巡逻|降落|起飞|返回|或", object_name):
            self.logger.info(f"靠近硬规则放行: 目标含动作碎片 '{object_name}'，交由LLM")
            return None
        # 如果目标名包含逗号/句号等标点，说明是截断碎片
        if re.search(r"[，,。；;]", object_name):
            self.logger.info(f"靠近硬规则放行: 目标含标点碎片 '{object_name}'，交由LLM")
            return None
        # 如果目标名过长（超过10字符），极可能是误提取的指令片段
        if len(object_name) > 10:
            self.logger.info(f"靠近硬规则放行: 目标名过长 '{object_name}'，交由LLM")
            return None

        parsed = {"type": "approach", "object_name": object_name}
        if distance_m is not None:
            parsed["distance_m"] = float(distance_m)
            parsed["modifier"] = modifier_key
        return parsed

    def _parse_return_home_clause(self, clause: str):
        """
        解析回到起飞点子句：回到起点/返回起飞点/回家/回起飞位置 等。
        返回 True 表示命中，False 表示未命中。
        """
        if not clause:
            return False
        text = re.sub(r"\s+", "", clause)
        return bool(re.search(
            r"回到起点|回起点|返回起点|返回起飞|回到起飞|回到出发|返回出发|回家|回到原点|返回原点",
            text,
        ))

    def _match_distance_modifier(self, text: str):
        mapping = getattr(self, "semantic_distance_map", {}) or {}
        if not mapping or not text:
            return "", None
        for key in sorted(mapping.keys(), key=len, reverse=True):
            if key and key in text:
                try:
                    return key, float(mapping[key])
                except Exception:
                    return key, None
        return "", None

    def _parse_search_clause(self, clause: str):
        """
        解析搜索子句：
        - 搜索/搜寻/查找某物体 -> quick模式
        - 周围所有某物体 -> all模式
        """
        if not clause:
            return None

        text = re.sub(r"\s+", "", clause)
        if not text:
            return None

        # 全景搜索模式：周围所有/附近所有/所有...（带周围语义）
        all_mode = False
        object_name = ""
        if re.search(r"(?:周围|附近).*(?:所有|全部)|(?:所有|全部).*(?:周围|附近)", text):
            all_mode = True
            # 提取“所有”后的对象描述
            m = re.search(r"(?:周围|附近)?(?:所有|全部)(.+?)$", text)
            object_name = (m.group(1) if m else "").strip()
        elif text.startswith("周围所有") or text.startswith("附近所有"):
            all_mode = True
            object_name = text.replace("周围所有", "", 1).replace("附近所有", "", 1).strip()

        if all_mode:
            object_name = re.sub(r"^(?:的|有)", "", object_name).strip()
            object_name = re.sub(r"(?:情况|有哪些|有什么|在哪|位置)$", "", object_name).strip()
            if not object_name:
                return None
            return {"type": "search", "mode": "all", "object_name": object_name}

        # 快速搜索模式
        m = re.search(r"(?:搜索|搜寻|查找|找到|找)(.+?)$", text)
        if not m:
            return None

        raw_tail = (m.group(1) or "").strip()

        # 如果"找到XX"后面还跟着动作动词（靠近/接近/飞向等），
        # 说明这是复合指令（如"找到红色气球靠近它"），不应被搜索硬规则独占，
        # 交给LLM整体处理
        if re.search(r"靠近|接近|飞向", raw_tail):
            self.logger.info(f"搜索硬规则放行: 检测到后续动作动词 '{raw_tail}'")
            return None

        object_name = raw_tail
        object_name = re.sub(r"^(?:一下|下|一个|目标|物体)", "", object_name).strip()
        object_name = re.sub(r"^到", "", object_name).strip()
        object_name = re.sub(r"(?:目标|物体|在哪里|在哪)$", "", object_name).strip()
        if not object_name:
            return None
        return {"type": "search", "mode": "quick", "object_name": object_name}

    def _execute_search_template(self, object_name: str, mode: str):
        """
        执行搜索模板：
        - quick: 当前视野优先，未命中再旋转，命中首个即结束
        - all: 旋转一圈，统计总数与相对朝向
        搜索未命中属于正常业务结果，不作为执行异常。
        """
        if self.search_object_function is None:
            print("执行失败：未注入search_object功能")
            self.logger.warning("搜索模板拒绝执行: 未注入search_object_function")
            return False

        try:
            # 归一化目标名（如"最近的红色气球" → "red balloon"）
            object_name = self._normalize_object_alias(object_name)
            # 兼容旧签名（仅object_name）与新签名（object_name, mode）
            try:
                found = self.search_object_function(object_name, mode=mode)
            except TypeError:
                found = self.search_object_function(object_name)

            summary = self._get_latest_result_cn(default_text="搜索完成")
            print(summary)
            self.logger.info(f"template_search mode={mode} target={object_name} found={bool(found)} summary={summary}")
            return True
        except Exception as e:
            print(f"执行失败：搜索模板异常 {e}")
            self.logger.error(f"搜索模板执行失败: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def _execute_approach_template(self, object_name: str, spatial_hint: str = ""):
        """
        执行靠近模板：搜索并持续逼近目标，直到满足停止条件。
        """
        if self.approachObjective_function is None:
            print("执行失败：未注入approach_objective功能")
            self.logger.warning("靠近模板拒绝执行: 未注入approachObjective_function")
            return False

        try:
            # 归一化目标名（如"最近的红色气球" → "red balloon"）
            object_name = self._normalize_object_alias(object_name)
            # 硬规则直接调用底层提供的全自动寻找+靠近高级函数，而不用暴露给LLM的基础控制函数
            comm_obj = getattr(self.search_object_function, "__self__", None)
            if not comm_obj or not hasattr(comm_obj, "approach_objective_to_target"):
                print("执行失败：底层未提供高级approach_objective_to_target功能")
                return False

            ok = comm_obj.approach_objective_to_target(object_name, spatial_hint=spatial_hint)
            summary = self._get_latest_result_cn(default_text="靠近完成")
            print(summary)
            self.logger.info(f"template_approach target={object_name} ok={bool(ok)} summary={summary}")
            return bool(ok)
        except Exception as e:
            print(f"执行失败：靠近模板异常 {e}")
            self.logger.error(f"靠近模板执行失败: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def _execute_approach_distance_template(self, object_name: str, distance_m: float):
        """
        执行“靠近一点点”模板：先对准目标，再前移固定距离。
        """
        if distance_m <= 0:
            print("执行失败：靠近距离必须大于0米")
            self.logger.warning(f"靠近一点点拒绝执行: distance={distance_m}")
            return False

        try:
            comm_obj = getattr(self.search_object_function, "__self__", None)
            if comm_obj is None or not hasattr(comm_obj, "face_objective_to_target"):
                print("执行失败：未注入face_objective_to_target功能")
                self.logger.warning("靠近一点点拒绝执行: 缺少face_objective_to_target")
                return False

            # 归一化目标名
            object_name = self._normalize_object_alias(object_name)
            ok = comm_obj.face_objective_to_target(object_name)
            if not ok:
                self.logger.warning(f"靠近一点点失败: 未能对准 {object_name}")
                return False

            moved = self._execute_body_move_template(
                dx_body=distance_m,
                dy_body=0.0,
                dz_body=0.0,
                distance_m=distance_m,
                direction_text=f"forward({distance_m:.2f}m)",
            )
            return bool(moved)
        except Exception as e:
            print(f"执行失败：靠近一点点异常 {e}")
            self.logger.error(f"靠近一点点执行失败: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def _return_home_and_land(self):
        comm_obj = getattr(self.search_object_function, "__self__", None)
        if comm_obj is None:
            self.logger.warning("返航降落失败: 未获取通信对象")
            return False

        home = getattr(comm_obj, "_home_pos_ned", None)
        if home is None or len(home) < 3:
            self.logger.warning("返航降落失败: 未找到home位置")
            return False

        try:
            mav = self.MavList[0]
            tx, ty, tz = float(home[0]), float(home[1]), float(home[2])
            yaw = float(mav.uavAngEular[2])
            mav.SendPosNED(tx, ty, tz, yaw)
            reached = self._wait_until_position_reached(tx, ty, tz, timeout_s=12.0, pos_tol=0.25)
            if not reached:
                self.logger.warning("返航降落失败: 未能到达home")
                return False

            land_fn = getattr(mav, "sendMavLand", None)
            if callable(land_fn):
                land_fn(tx, ty, tz)
                return True
            self.logger.warning("返航降落失败: sendMavLand不可用")
            return False
        except Exception as e:
            self.logger.error(f"返航降落异常: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def _handle_conditional_task(self, task: str):
        """
        识别“有的话/没有就/否则”的条件指令并执行分支。
        例如：先巡逻一下四周，看看有没有红色气球，有的话就靠近着它，没有就降落
        """
        if not task:
            return False, False, ""

        text = re.sub(r"\s+", "", task)
        if not re.search(r"(?:有没有|如果有).*(?:有的话|没有就|否则)", text):
            return False, False, ""

        m = re.search(r"(?:有没有|如果有)(.+?)(?:有的话|就|则|否则|没有|无|,|，|。)", text)
        if not m:
            return False, False, ""

        object_name = (m.group(1) or "").strip()
        object_name = re.sub(r"^(?:的|个|一个)", "", object_name).strip()
        object_name = re.sub(r"(?:目标|物体)$", "", object_name).strip()
        if not object_name:
            return False, False, ""

        search_mode = "all" if re.search(r"(巡逻|四周|全景)", text) else "quick"

        try:
            try:
                found = self.search_object_function(object_name, mode=search_mode)
            except TypeError:
                found = self.search_object_function(object_name)
        except Exception as e:
            self.logger.warning(f"条件模板搜索异常: {e}")
            return True, False, "条件模板搜索失败"

        if found:
            _, distance_m = self._match_distance_modifier(text)
            if distance_m is not None:
                ok = self._execute_approach_distance_template(object_name, distance_m)
            else:
                ok = self._execute_approach_template(object_name, spatial_hint=text)
            summary = self._get_latest_result_cn(default_text="靠近完成") if ok else "靠近失败"
            return True, bool(ok), summary

        ok = self._return_home_and_land()
        summary = "未发现目标，已返航降落" if ok else "未发现目标，返航降落失败"
        return True, bool(ok), summary

    def _execute_body_move_template(self, dx_body: float, dy_body: float, dz_body: float, distance_m: float, direction_text: str):
        """
        固定位移模板：机头为前，统一通过b2n将机体系位移转换到NED后发送位置指令。
        失败时不发送任何控制命令。
        """
        if distance_m <= 0:
            print("执行失败：位移距离必须大于0米")
            self.logger.warning(f"模板位移拒绝执行: 非法距离 distance={distance_m}")
            return False
        if distance_m > 10:
            print("执行失败：单次位移不能超过10米")
            self.logger.warning(f"模板位移拒绝执行: 超出上限 distance={distance_m}")
            return False

        try:
            mav = self.MavList[0]
            roll, pitch, yaw = mav.uavAngEular[0], mav.uavAngEular[1], mav.uavAngEular[2]
            cur_x, cur_y, cur_z = mav.uavPosNED[0], mav.uavPosNED[1], mav.uavPosNED[2]

            dx_ned, dy_ned, dz_ned = b2n(dx_body, dy_body, dz_body, roll, pitch, yaw)
            target_x = cur_x + dx_ned
            target_y = cur_y + dy_ned
            target_z = cur_z + dz_ned

            # 精简日志：仅打印关键参数
            print(
                f"[MOVE] dir={direction_text} dist={distance_m:.2f} yaw={yaw:.3f} "
                f"d_body=({dx_body:.2f},{dy_body:.2f},{dz_body:.2f}) "
                f"d_ned=({dx_ned:.2f},{dy_ned:.2f},{dz_ned:.2f}) "
                f"target=({target_x:.2f},{target_y:.2f},{target_z:.2f})"
            )
            self.logger.info(
                f"template_move dir={direction_text} dist={distance_m:.2f} yaw={yaw:.3f} "
                f"d_body=({dx_body:.2f},{dy_body:.2f},{dz_body:.2f}) "
                f"d_ned=({dx_ned:.2f},{dy_ned:.2f},{dz_ned:.2f}) "
                f"target=({target_x:.2f},{target_y:.2f},{target_z:.2f})"
            )

            mav.SendPosNED(target_x, target_y, target_z, yaw)
            reached = self._wait_until_position_reached(target_x, target_y, target_z, timeout_s=12.0, pos_tol=0.18)
            if not reached:
                print("执行失败：位移在超时时间内未到位，已中止后续子句")
                self.logger.warning("模板位移超时未到位，终止后续子句")
                return False
            return True
        except Exception as e:
            print(f"执行失败：模板位移异常 {e}")
            self.logger.error(f"模板位移执行失败: {e}")
            self.logger.debug(traceback.format_exc())
            # 失败即不发送任何控制命令：异常情况下直接返回
            return False

    def _execute_turn_template(self, sign: float, deg: float, turn_text: str):
        """
        固定转向模板：保持当前位置，仅改变yaw。
        约定：左转为负，右转为正，和当前模拟器表现保持一致。
        """
        if deg <= 0:
            print("执行失败：转向角度必须大于0度")
            self.logger.warning(f"模板转向拒绝执行: 非法角度 deg={deg}")
            return False
        if deg > 180:
            print("执行失败：单次转向不能超过180度")
            self.logger.warning(f"模板转向拒绝执行: 超出上限 deg={deg}")
            return False

        try:
            mav = self.MavList[0]
            cur_x, cur_y, cur_z = mav.uavPosNED[0], mav.uavPosNED[1], mav.uavPosNED[2]
            cur_yaw = mav.uavAngEular[2]
            delta = math.radians(deg) * sign
            target_yaw = cur_yaw + delta

            print(
                f"[TURN] dir={turn_text} deg={deg:.2f} yaw={cur_yaw:.3f}-> {target_yaw:.3f}"
            )
            self.logger.info(
                f"template_turn dir={turn_text} deg={deg:.2f} yaw={cur_yaw:.3f}->{target_yaw:.3f}"
            )

            mav.SendPosNED(cur_x, cur_y, cur_z, target_yaw)
            reached = self._wait_until_yaw_reached(target_yaw, timeout_s=8.0, yaw_tol_deg=4.0)
            if not reached:
                print("执行失败：转向在超时时间内未到位，已中止后续子句")
                self.logger.warning("模板转向超时未到位，终止后续子句")
                return False
            return True
        except Exception as e:
            print(f"执行失败：模板转向异常 {e}")
            self.logger.error(f"模板转向执行失败: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def _wait_until_position_reached(self, tx: float, ty: float, tz: float, timeout_s: float = 12.0, pos_tol: float = 0.18):
        """
        等待无人机位置到达目标点，避免后续子句提前执行。
        如果目标高度触发了安全保护，自动调整目标高度以避免等待超时。
        """
        try:
            comm_obj = getattr(self.search_object_function, "__self__", None)
            if comm_obj is not None and hasattr(comm_obj, "get_safety_summary"):
                safety = comm_obj.get_safety_summary()
                if safety.get("enable_alt_guard", True):
                    floor = safety.get("alt_floor_ned", -0.3)
                    ceiling = safety.get("alt_ceiling_ned", -5.0)
                    safe_alt = safety.get("alt_safe_ned", -0.5)
                    if tz > floor:
                        self.logger.info(f"wait_pos: 目标高度 tz={tz:.2f} 触发下限保护 {floor:.2f}，期望调整为修正高度 {safe_alt:.2f}")
                        tz = safe_alt
                    elif tz < ceiling:
                        self.logger.info(f"wait_pos: 目标高度 tz={tz:.2f} 触发上限保护 {ceiling:.2f}，期望调整为限制高度 {ceiling:.2f}")
                        tz = ceiling
        except Exception as e:
            self.logger.debug(f"检查安全保护配置失败: {e}")
        start = time.monotonic()
        while time.monotonic() - start <= timeout_s:
            x, y, z = self.MavList[0].uavPosNED[0], self.MavList[0].uavPosNED[1], self.MavList[0].uavPosNED[2]
            dist = math.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
            if dist <= pos_tol:
                self.logger.info(f"move_reached dist={dist:.3f} tol={pos_tol:.3f}")
                return True
            time.sleep(0.05)

        x, y, z = self.MavList[0].uavPosNED[0], self.MavList[0].uavPosNED[1], self.MavList[0].uavPosNED[2]
        dist = math.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
        self.logger.warning(
            f"move_timeout dist={dist:.3f} tol={pos_tol:.3f} target=({tx:.2f},{ty:.2f},{tz:.2f}) current=({x:.2f},{y:.2f},{z:.2f})"
        )
        return False

    def _wait_until_yaw_reached(self, target_yaw: float, timeout_s: float = 8.0, yaw_tol_deg: float = 4.0):
        """
        等待无人机偏航到达目标角。
        """
        tol = math.radians(yaw_tol_deg)
        start = time.monotonic()
        while time.monotonic() - start <= timeout_s:
            yaw = self.MavList[0].uavAngEular[2]
            err = abs(self._angle_diff(target_yaw, yaw))
            if err <= tol:
                self.logger.info(f"turn_reached err_deg={math.degrees(err):.2f} tol_deg={yaw_tol_deg:.2f}")
                return True
            time.sleep(0.05)

        yaw = self.MavList[0].uavAngEular[2]
        err = abs(self._angle_diff(target_yaw, yaw))
        self.logger.warning(
            f"turn_timeout err_deg={math.degrees(err):.2f} tol_deg={yaw_tol_deg:.2f} target_yaw={target_yaw:.3f} current_yaw={yaw:.3f}"
        )
        return False

    @staticmethod
    def _angle_diff(target: float, current: float):
        """
        计算[-pi, pi]范围内的角度差。
        """
        d = target - current
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    def _execute_face_object_template(self, object_name: str):
        """
        原地朝向目标，不前进。
        """
        if self.face_objective_function is None:
            print("执行失败：未注入face_objective功能")
            self.logger.warning("原地朝向目标拒绝执行: 未注入face_objective_function")
            return False
        # 归一化目标名
        object_name = self._normalize_object_alias(object_name)
        ok = self.face_objective_function(object_name)
        if not ok:
            self.logger.warning(f"原地朝向目标失败: {object_name}")
            return False
        return True

    def _execute_return_home_template(self):
        """
        飞回起飞点（_home_pos_ned），分两段飞行避免冲过头，
        最后校正朝向到起飞时的方向。不执行降落，仅悬停在起飞点上方。
        """
        try:
            comm_obj = getattr(self.search_object_function, "__self__", None)
            home = None
            if comm_obj is not None:
                home = getattr(comm_obj, "_home_pos_ned", None)
            if home is None or len(home) < 3:
                print("执行失败：未找到起飞点位置")
                self.logger.warning("返回起飞点失败: 未找到_home_pos_ned")
                return False

            mav = self.MavList[0]
            tx, ty, tz = float(home[0]), float(home[1]), float(home[2])
            cur_x = float(mav.uavPosNED[0])
            cur_y = float(mav.uavPosNED[1])
            cur_z = float(mav.uavPosNED[2])
            yaw = float(mav.uavAngEular[2])
            home_yaw = getattr(comm_obj, "_home_yaw", yaw) if comm_obj else yaw

            # 预处理：如果贴地（高度 > -0.3m），先爬升到安全高度再水平移动
            safe_cruise_alt = -0.5
            if cur_z > -0.3:
                self.logger.info(
                    f"return_home pre-climb: 贴地修正 alt={cur_z:.2f} -> {safe_cruise_alt:.2f}"
                )
                mav.SendPosNED(cur_x, cur_y, safe_cruise_alt, yaw)
                self._wait_until_position_reached(
                    cur_x, cur_y, safe_cruise_alt, timeout_s=5.0, pos_tol=0.25
                )
                cur_z = safe_cruise_alt

            # 计算与起飞点的水平距离
            dx = tx - cur_x
            dy = ty - cur_y
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > 1.0:
                # Phase1a: 先飞到中点，减速停稳
                mid_x = cur_x + dx * 0.5
                mid_y = cur_y + dy * 0.5
                self.logger.info(
                    f"return_home phase1a: fly to midpoint ({mid_x:.2f},{mid_y:.2f},{cur_z:.2f})"
                )
                mav.SendPosNED(mid_x, mid_y, cur_z, yaw)
                self._wait_until_position_reached(
                    mid_x, mid_y, cur_z, timeout_s=15.0, pos_tol=0.25
                )
                mav.SendVelFRD(0, 0, 0, 0)
                time.sleep(0.3)

                # Phase1b: 从中点飞到起飞点正上方
                self.logger.info(
                    f"return_home phase1b: fly to home ({tx:.2f},{ty:.2f},{cur_z:.2f})"
                )
                mav.SendPosNED(tx, ty, cur_z, yaw)
                reached_xy = self._wait_until_position_reached(
                    tx, ty, cur_z, timeout_s=15.0, pos_tol=0.25
                )
            else:
                self.logger.info(
                    f"return_home phase1: fly to ({tx:.2f},{ty:.2f},{cur_z:.2f})"
                )
                mav.SendPosNED(tx, ty, cur_z, yaw)
                reached_xy = self._wait_until_position_reached(
                    tx, ty, cur_z, timeout_s=15.0, pos_tol=0.25
                )

            if not reached_xy:
                print("执行失败：返回起飞点水平位移超时")
                self.logger.warning("返回起飞点失败: 水平位移超时")
                return False

            # Phase2: 调整到起飞高度
            if abs(cur_z - tz) > 0.15:
                self.logger.info(
                    f"return_home phase2: adjust alt to {tz:.2f}"
                )
                mav.SendPosNED(tx, ty, tz, home_yaw)
                reached_z = self._wait_until_position_reached(
                    tx, ty, tz, timeout_s=8.0, pos_tol=0.25
                )
                if not reached_z:
                    self.logger.warning("返回起飞点: 高度调整超时，已到达水平位置")

            # Phase3: 校正朝向到起飞时的方向
            self.logger.info(f"return_home phase3: correct yaw {yaw:.3f} -> {home_yaw:.3f}")
            mav.SendPosNED(tx, ty, tz, home_yaw)
            time.sleep(1.0)

            print("已返回起飞点")
            self.logger.info("return_home 完成")
            return True
        except Exception as e:
            print(f"执行失败：返回起飞点异常 {e}")
            self.logger.error(f"返回起飞点异常: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def _run_agent_for_clause(self, agent, clause: str):
        """
        执行单个非模板子句（展示模式：仅单次生成，不走SmolAgents内部执行）。
        """
        # 清理上一任务残留摘要，避免本轮结果展示被历史搜索信息污染。
        try:
            comm_obj = getattr(self.search_object_function, "__self__", None)
            if comm_obj is not None and hasattr(comm_obj, "last_search_result_cn"):
                comm_obj.last_search_result_cn = ""
        except Exception:
            pass

        start_time = time.time()
        self.logger.info("本轮请求模式=单次生成")
        self.logger.info(f"步骤执行方式=AI生成 子句={clause}")
        self.logger.info(f"开始请求模型生成代码, 任务={clause}")

        # 直接调用底层模型进行单次代码生成，避免SmolAgents内部解释执行带来的上下文不一致。
        class _Msg:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        messages = [
            _Msg("system", self.Prompt_dit["Prompt_smol"]),
            _Msg("user", clause),
        ]

        code = ""
        try:
            resp = agent.model.generate(messages)
            code = getattr(resp, "content", "") or ""
            print(code)
        except Exception as e:
            print(f"模型生成失败：{e}")
            self.logger.error(f"模型生成失败: {e}")
            self.logger.debug(traceback.format_exc())
            return False

        self.logger.info("模型代码生成完成")
        print("AI计算时间：", time.time() - start_time, "s")
        self.logger.info(f"AI计算时间: {time.time() - start_time:.3f}s")

        if code.strip():
            self._task_guard_state = {
                "deadline": time.monotonic() + 180.0,
                "search_calls": 0,
                "max_search_calls": 4,
                "clause": clause,
            }
            ok = self.execute_generated_code(code)
            self._task_guard_state = None
            return bool(ok)
        self.logger.warning("本轮未收到可执行代码")
        print("未生成可执行代码，请重试指令。")
        return False

    # 记录聊天历史记录
    def GetHistrory(self, prompt, Answer):
        # 获取当前UTC时间
        UTCTime = datetime.now(timezone.utc)
        # 格式化时间
        TimeTemp = UTCTime.strftime("%Y-%m-%d %H:%M:%S %Z")
        # 将时间、问题和回答记录到聊天历史中
        self.chatHistory.append({"Time": TimeTemp, "Qustion": prompt, "Answer": Answer})

        # 执行生成的代码

    def execute_generated_code(self, code: str):
        # 每次执行都重置搜索计数器，防止 test_exp2.py 直接调用时计数器跨指令累积
        self._task_guard_state = {
            "deadline": time.monotonic() + 180.0,
            "search_calls": 0,
            "max_search_calls": 4,
            "clause": "",
        }
        # 定义全局命名空间，包含当前类实例、time模块、body_to_ned函数和final_answer函数
        exec_globals = {
            "self": self,
            "time": time,
            "math": math,
            "b2n": b2n,
            "display": lambda *args, **kwargs: None,
            "final_answer": lambda x: print(f"执行成功：{x}"),
        }
        # 兼容<code>和```python两种代码包裹格式
        code = code.strip()
        code_tag_match = re.search(r"<code>\s*([\s\S]*?)\s*</code>", code, flags=re.IGNORECASE)
        if code_tag_match:
            code = code_tag_match.group(1).strip()
        fence_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", code, flags=re.IGNORECASE)
        clean_code = fence_match.group(1).strip() if fence_match else code
        # 清理模型常见误导入：b2n已由执行器注入，不需要from utils import b2n
        clean_code = re.sub(r"^\s*from\s+utils\s+import\s+b2n\s*$", "", clean_code, flags=re.MULTILINE)
        # 兼容可视化语句：display已注入为no-op，无需依赖IPython。
        clean_code = re.sub(r"^\s*from\s+IPython\.display\s+import\s+display\s*$", "", clean_code, flags=re.MULTILINE)
        # 仅在已执行过场外模式启动后，移除生成代码中的启动片段，避免重复 initOffboard。
        if getattr(self, "_init_sequence_done", False):
            try:
                lines = clean_code.splitlines()
                new_lines = []
                skipping = False
                for line in lines:
                    if "initOffboard" in line or "# 启动场外模式" in line:
                        skipping = True
                        continue
                    if skipping and ("SendPosNED" in line or "time.sleep" in line or line.strip() == ""):
                        continue
                    if skipping:
                        skipping = False
                    new_lines.append(line)
                clean_code = "\n".join(new_lines)
            except Exception:
                pass
        # 为LLM生成代码注入受限工具包装器，避免参数污染与无限搜索循环。
        # 仅在首次调用时保存原始（未包装）函数引用，防止多次调用导致包装器嵌套叠加。
        if not hasattr(self, "_orig_search_function"):
            self._orig_search_function = self.search_object_function
            self._orig_approach_function = self.approachObjective_function
            self._orig_detect_function = self.detect_function
            self._orig_face_function = self.face_objective_function
            self._orig_strike_function = self.strike_objective_function
        orig_search = self._orig_search_function
        orig_approach = self._orig_approach_function
        orig_detect = self._orig_detect_function
        orig_face = self._orig_face_function
        orig_strike = self._orig_strike_function

        class SearchResult:
            def __init__(self, found, obj_list=None, obj_locs=None, obj_logits=None, img_with_box=None, summary=None):
                self.found = bool(found)
                self.obj_list = obj_list or []
                self.obj_locs = obj_locs or []
                self.obj_logits = obj_logits or []
                self.img_with_box = img_with_box
                self.summary = summary

            def __iter__(self):
                yield self.found
                yield self.obj_list
                yield self.obj_locs
                yield self.obj_logits
                yield self.img_with_box

            def __bool__(self):
                return self.found

        def safe_search(target, mode="quick"):
            self._guard_check_deadline()
            valid = self._validate_target_name(target)
            if valid is None:
                raise ValueError(f"非法目标名称: {target}")
            valid = self._normalize_object_alias(valid)
            state = getattr(self, "_task_guard_state", None) or {}
            state["search_calls"] = int(state.get("search_calls", 0)) + 1
            if state["search_calls"] > int(state.get("max_search_calls", 4)):
                try:
                    self.MavList[0].SendVelFRD(0, 0, 0, 0)
                except Exception:
                    pass
                raise RuntimeError("搜索重试次数超限，已中止任务")
            self._task_guard_state = state
            found = orig_search(valid, mode=mode)

            obj_list = []
            obj_locs = []
            obj_logits = []
            img_with_box = None
            if callable(getattr(self, "detect_function", None)):
                try:
                    detect_result = self.detect_function(valid)
                    if isinstance(detect_result, dict):
                        obj_list = detect_result.get("obj_list", []) or []
                        obj_locs = detect_result.get("obj_locs", []) or []
                        obj_logits = detect_result.get("obj_logits", []) or []
                        img_with_box = detect_result.get("img_with_box")
                    elif isinstance(detect_result, (list, tuple)) and len(detect_result) >= 4:
                        obj_list, obj_locs, obj_logits, img_with_box = detect_result[:4]
                except Exception:
                    self.logger.debug("safe_search: 补充检测结果失败", exc_info=True)

            summary = self._get_latest_result_cn(default_text="搜索完成")
            return SearchResult(found, obj_list, obj_locs, obj_logits, img_with_box, summary=summary)

        def safe_approach(*args):
            self._guard_check_deadline()
            if len(args) == 2:
                error_x, error_y = args
                if not isinstance(error_x, (int, float)) or not isinstance(error_y, (int, float)):
                    raise ValueError(f"非法误差参数: {args}")
                return orig_approach(float(error_x), float(error_y))
            if len(args) == 1:
                target = args[0]
                valid = self._validate_target_name(target)
                if valid is None:
                    raise ValueError(f"非法目标名称: {target}")
                valid = self._normalize_object_alias(valid)
                # 兼容旧代码：如果LLM仍传目标名，则把它转回高层靠近目标接口
                if hasattr(self, "approach_objective_function") and callable(getattr(self, "approach_objective_function")):
                    return self.approach_objective_function(valid)
                return orig_approach(valid)
            raise ValueError(f"靠近参数数量错误: {args}")

        def safe_face(target):
            self._guard_check_deadline()
            valid = self._validate_target_name(target)
            if valid is None:
                raise ValueError(f"非法目标名称: {target}")
            valid = self._normalize_object_alias(valid)
            if orig_face is None:
                raise RuntimeError("face_objective_function未注入")
            return orig_face(valid)

        def safe_strike(target):
            self._guard_check_deadline()
            valid = self._validate_target_name(target)
            if valid is None:
                raise ValueError(f"非法目标名称: {target}")
            valid = self._normalize_object_alias(valid)
            if orig_strike is None:
                raise RuntimeError("strike_objective_function未注入")
            return orig_strike(valid)

        def guard_offboard_init(mav):
            """
            将 initOffboard 包装为幂等调用，避免同一控制对象重复启动线程。
            """
            if mav is None:
                return
            if getattr(mav, "_copilot_init_offboard_guarded", False):
                return

            original_init_offboard = getattr(mav, "initOffboard", None)
            if not callable(original_init_offboard):
                return

            state_attr = "_copilot_offboard_started"
            if not hasattr(mav, state_attr):
                setattr(mav, state_attr, False)

            def _guarded_init_offboard(*args, **kwargs):
                if getattr(mav, state_attr, False):
                    self.logger.info("initOffboard 已处于启动状态，跳过重复调用")
                    return True
                result = original_init_offboard(*args, **kwargs)
                setattr(mav, state_attr, True)
                return result

            mav.initOffboard = _guarded_init_offboard
            mav._copilot_init_offboard_guarded = True

        for mav in getattr(self, "MavList", []) or []:
            try:
                guard_offboard_init(mav)
            except Exception:
                self.logger.debug("initOffboard 幂等包装失败", exc_info=True)

        def safe_detect(target):
            if target is None or str(target).strip() == "":
                return orig_detect("")
            valid = self._validate_target_name(target)
            if valid is None:
                raise ValueError(f"非法目标名称: {target}")
            canonical = self._normalize_object_alias(valid)
            result = orig_detect(canonical)
            if isinstance(result, tuple) and len(result) >= 4:
                obj_list = list(result[0]) if result[0] is not None else []
                obj_locs = result[1]
                obj_logits = result[2]
                img_with_box = result[3]
                if canonical not in obj_list and valid not in obj_list:
                    return result
                if valid not in obj_list:
                    obj_list = [valid] + obj_list
                return obj_list, obj_locs, obj_logits, img_with_box
            return result

        self.search_object_function = safe_search
        self.approachObjective_function = safe_approach
        self.detect_function = safe_detect
        self.face_objective_function = safe_face
        self.strike_objective_function = safe_strike

        # 兼容模型误写的 `from utils import b2n`。
        utils_backup = sys.modules.get("utils")
        shim_utils = types.ModuleType("utils")
        shim_utils.b2n = b2n
        sys.modules["utils"] = shim_utils

        try:
            # 在执行前保存生成的代码到 logs/code 目录，文件仅包含代码
            try:
                # 如果当前正在处理某个cmd，则使用 cmd id 的统一文件名（追加），否则按时间新建
                logs_code_dir = os.path.join(os.path.dirname(__file__), "logs", "code")
                os.makedirs(logs_code_dir, exist_ok=True)
                if getattr(self, '_current_cmd_id', None):
                    file_name = f"{self._current_cmd_id}.py"
                    code_path = os.path.join(logs_code_dir, file_name)
                    header = f"\n# --- Generated snippet at {datetime.now().isoformat()} ---\n"
                    with open(code_path, "a", encoding="utf-8") as cf:
                        cf.write(header)
                        cf.write(clean_code)
                        cf.write("\n\n")
                else:
                    file_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".py"
                    code_path = os.path.join(logs_code_dir, file_name)
                    with open(code_path, "w", encoding="utf-8") as cf:
                        cf.write(clean_code)
                self._last_generated_code_path = code_path
                self.logger.info(f"保存生成代码: {code_path}")
            except Exception as e:
                self.logger.warning(f"保存生成代码失败: {e}")

            self.logger.info("开始执行生成代码")
            self.logger.debug(f"代码内容:\n{clean_code}")
            
            # --- 仅首次保留场外模式启动序列逻辑 ---
            flag_set_init_done = False
            try:
                lines = clean_code.splitlines()
                top_chunk = "\n".join(lines[:min(10, len(lines))])
                has_init_block = '# 启动场外模式' in top_chunk or ('initOffboard' in top_chunk and 'SendPosNED' in top_chunk)
                
                if has_init_block and getattr(self, "_init_sequence_done", False):
                    new_lines = []
                    i = 0
                    skip_until_endblock = False
                    while i < len(lines):
                        line = lines[i]
                        if i < 10 and ('# 启动场外模式' in line or 'initOffboard' in line):
                            skip_until_endblock = True
                            i += 1
                            continue
                        if skip_until_endblock and (i < 10) and ('SendPosNED' in line or 'time.sleep' in line or line.strip() == ''):
                            i += 1
                            continue
                        if skip_until_endblock and line.strip() and i >= 5:
                            skip_until_endblock = False
                        new_lines.append(line)
                        i += 1
                    clean_code = "\n".join(new_lines)
                    self.logger.info("已移除启动序列（已在首次执行过）")
                elif has_init_block and not getattr(self, "_init_sequence_done", False):
                    flag_set_init_done = True
                    self.logger.info("首次执行启动序列，后续将跳过此段")
            except Exception as e:
                self.logger.warning(f"启动序列处理失败: {e}")
                flag_set_init_done = False
            # --- end 仅首次保留逻辑 ---
            
            clean_exec_code = clean_code

            # 执行经消毒的代码
            exec(clean_exec_code, exec_globals)
            self.logger.info("生成代码执行完成")
            
            # 如果此次执行包含首次场外模式启动序列，则设置已执行标志
            try:
                if flag_set_init_done:
                    self._init_sequence_done = True
                    self.logger.info("场外模式启动序列已执行，后续代码将自动跳过此段")
            except Exception:
                pass
            
            return True
        except Exception as e:
            # 捕获并打印执行过程中可能出现的异常
            print(f"执行失败：{e}")
            self.logger.error(f"生成代码执行失败: {e}")
            self.logger.debug(traceback.format_exc())
            return False
        finally:
            # ── 无论成功/异常，都强制零速悬停，防止 LLM 代码残余速度 ──
            try:
                self.MavList[0].SendVelFRD(0, 0, 0, 0)
            except Exception:
                pass
            self.search_object_function = orig_search
            self.approachObjective_function = orig_approach
            self.detect_function = orig_detect
            self.face_objective_function = orig_face
            self.strike_objective_function = orig_strike
            if utils_backup is None:
                sys.modules.pop("utils", None)
            else:
                sys.modules["utils"] = utils_backup

    # 智能体模式
    def Agents_UAV(self):
        # 定义提示模板
        prompt_templates = PromptTemplates(
            system_prompt=self.Prompt_dit["Prompt_smol"],
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

        # ── 双线程启动 ────────────────────────────────────────────────────
        # 重置终止信号，防止重复调用时残留
        self._stop_signal = False
        self.is_interrupted = False

        watchdog = threading.Thread(
            target=self._watchdog_input_loop,
            args=(agent,),
            name="Watchdog-Input",
            daemon=True,
        )
        worker = threading.Thread(
            target=self._agent_worker_loop,
            args=(agent,),
            name="Agent-Worker",
            daemon=True,
        )
        watchdog.start()
        worker.start()
        self.logger.info("双线程已启动: Watchdog-Input + Agent-Worker")
        print("\n系统已就绪（双线程模式）：随时可输入指令，急停指令将被立即响应。\n")

        try:
            watchdog.join()
            worker.join()
        except KeyboardInterrupt:
            self.logger.info("收到 KeyboardInterrupt，正在关闭双线程...")
            self._stop_signal = True
            self._task_queue.put(None)
        # ─────────────────────────────────────────────────────────────────



    # 主控制逻辑
    def _emergency_stop(self):
        """
        紧急停止：速度清零 + 置中断标志 + 清空任务队列。
        可被看门狗线程和 Agent Worker 线程同时安全调用。
        """
        # 1. 向飞控发速度清零
        try:
            self.MavList[0].SendVelFRD(0, 0, 0, 0)
        except Exception as e:
            self.logger.error(f"紧急停止 SendVelFRD 失败: {e}")
        # 2. 置全局中断标志，通知所有耗时函数尽快退出
        self.is_interrupted = True
        # 3. 清空任务队列，丢弃所有待执行的常规任务
        try:
            while not self._task_queue.empty():
                self._task_queue.get_nowait()
                self._task_queue.task_done()
        except Exception:
            pass
        self.logger.warning("[WATCHDOG] 紧急停止触发: 速度清零 / 中断标志已置 / 任务队列已清空")
        print("\n⚠️  [紧急停止] 飞机已悬停，任务队列已清空。请输入新指令。")

    def _watchdog_input_loop(self, agent):
        """
        看门狗线程：负责接收用户输入。
        - 紧急指令 → 立即调用 _emergency_stop()，不入队。
        - 退出指令 → 设置终止信号并退出。
        - 常规指令 → 放入任务队列，由 Agent Worker 顺序执行。
        """
        self.logger.info("[WATCHDOG] 看门狗线程已启动，等待指令...")
        while not self._stop_signal:
            try:
                task = input("\n请输入你的控制模式指令: ").strip()
            except (EOFError, KeyboardInterrupt):
                self._stop_signal = True
                self._task_queue.put(None)  # 唤醒 worker 退出
                break

            if not task:
                print("指令不能为空，请重新输入！")
                continue

            # 退出指令
            if task.lower() in self.ExitList:
                print("对话结束，程序退出。")
                self.logger.info("[WATCHDOG] 用户主动退出")
                self._stop_signal = True
                self._task_queue.put(None)  # 唤醒 worker 退出
                break

            # 紧急停止检测（始终最高优先级，不入队）
            if re.search(
                r"急停|紧急停止|立即停止|快停|停下|停止|悬停|stop\b|halt\b",
                task, flags=re.IGNORECASE
            ):
                self.logger.warning(f"[WATCHDOG] 检测到紧急指令: '{task}'，立即执行急停")
                self._emergency_stop()
                continue

            # 常规指令：放入队列，清除上一次中断状态以便继续执行
            self.is_interrupted = False
            self._task_queue.put(task)
            self.logger.info(f"[WATCHDOG] 指令已入队: '{task}'")

    def _agent_worker_loop(self, agent):
        """
        Agent Worker 线程：从任务队列中顺序取指令并执行。
        收到 None 为退出信号。
        """
        self.logger.info("[WORKER] Agent Worker 线程已启动，等待任务...")
        while not self._stop_signal:
            try:
                task = self._task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:
                # 退出信号
                self._task_queue.task_done()
                break

            cmd_id = ""
            overall_ok = False
            last_summary = ""
            self.logger.info(f"[WORKER] 开始执行任务: '{task}'")
            try:
                cmd_start_time = time.time()
                cmd_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 推送简洁窗口：当前指令
                try:
                    if getattr(self, 'output_window', None) is not None:
                        self.output_window.show_command(task)
                except Exception:
                    pass

                handled, ok, summary = self._handle_conditional_task(task)
                if handled:
                    self._emit_step_result(cmd_id, 1, 1, "硬规则-条件", bool(ok), summary)
                    cmd_cost = time.time() - cmd_start_time
                    overall_ok = bool(ok)
                    last_summary = summary or ("执行完成" if ok else "执行失败")
                    if ok:
                        self._emit_highlight_block(
                            "任务结果",
                            [
                                f"任务编号: {cmd_id}",
                                "状态: 成功",
                                "执行方式: 条件分支模板",
                                f"关键结果: {summary or '执行完成'}",
                                f"总耗时: {cmd_cost:.2f} 秒",
                            ],
                            ok=True,
                        )
                    else:
                        self._emit_highlight_block(
                            "任务结果",
                            [
                                f"任务编号: {cmd_id}",
                                "状态: 失败",
                                "执行方式: 条件分支模板",
                                f"关键结果: {summary or '执行失败'}",
                                f"总耗时: {cmd_cost:.2f} 秒",
                            ],
                            ok=False,
                        )
                    continue

                if self._is_complex_instruction(task):
                    self.logger.info(f"复杂指令检测: 跳过子句拆分, 整条交给LLM task={task}")
                    clauses = [task]
                    _skip_hard_rules = True
                else:
                    clauses = self._split_task_clauses(task)
                    if not clauses:
                        clauses = [task]
                    _skip_hard_rules = False
                overall_ok = True
                last_summary = ""

                for idx, clause in enumerate(clauses, start=1):
                    # 每个子句执行前检查中断标志
                    if self.is_interrupted:
                        self.logger.warning(f"[WORKER] 任务被中断，跳过剩余子句 (clause={clause})")
                        overall_ok = False
                        last_summary = "任务被紧急中断"
                        break

                    self.logger.info(f"CLAUSE_START cmd_id={cmd_id} idx={idx}/{len(clauses)} clause={clause}")
                    self._reset_comm_task_timeout(clause)

                    # 复杂指令跳过硬规则，直接交给LLM
                    if not _skip_hard_rules:
                        action, summary = self._handle_hard_rules(clause)
                        if action == "continue":
                            step_ok = not any(k in (summary or "") for k in ("失败", "拒绝", "冲突", "超时", "异常"))
                            last_summary = summary
                            self._emit_step_result(cmd_id, idx, len(clauses), "硬规则", step_ok, summary)
                            if not step_ok:
                                overall_ok = False
                                self.logger.warning(f"CLAUSE_ABORT cmd_id={cmd_id} idx={idx} reason={summary}")
                                break
                            continue

                    ok = self._run_agent_for_clause(agent, clause)
                    if ok:
                        summary = self._get_latest_result_cn(default_text="执行完成")
                    else:
                        summary = "LLM生成执行失败"

                    # 若存在生成代码文件，解析并展示AI解析结果（尝试提取目标/动作/条件）
                    try:
                        code_path = getattr(self, '_last_generated_code_path', None)
                        if code_path and os.path.exists(code_path):
                            try:
                                with open(code_path, 'r', encoding='utf-8') as rf:
                                    txt = rf.read()
                            except Exception:
                                txt = ''
                            ai_parse = { '目标': '', '动作': '', '程度': '', '条件': '' }
                            # 尝试匹配中文关键词
                            m = re.search(r"目标[:：]\s*(.+)", txt)
                            if m:
                                ai_parse['目标'] = m.group(1).strip()
                            # 动作尝试从代码注释或函数名推断
                            act_m = re.search(r"(巡逻|搜索|靠近|降落|上升|前进|后退|move_with_speed|SendPosNED)", txt)
                            if act_m:
                                ai_parse['动作'] = act_m.group(1)
                            cond_m = re.search(r"如果|有的话|没有就|否则|未找到|找到", txt)
                            if cond_m:
                                ai_parse['条件'] = cond_m.group(0)
                            # 清除最近生成标记，避免误读后续
                            self._last_generated_code_path = None
                    except Exception:
                        pass
                    last_summary = summary
                    self._emit_step_result(cmd_id, idx, len(clauses), "LLM主导", bool(ok), summary)
                    if not ok:
                        overall_ok = False
                        self.logger.warning(f"CLAUSE_ABORT cmd_id={cmd_id} idx={idx} reason={summary}")
                        break

                cmd_cost = time.time() - cmd_start_time
                if overall_ok:
                    self._emit_highlight_block(
                        "任务结果",
                        [
                            f"任务编号: {cmd_id}",
                            "状态: 成功",
                            "执行方式: 多子句顺序执行",
                            f"关键结果: {last_summary or '执行完成'}",
                            f"总耗时: {cmd_cost:.2f} 秒",
                        ],
                        ok=True,
                    )
                else:
                    self._emit_highlight_block(
                        "任务结果",
                        [
                            f"任务编号: {cmd_id}",
                            "状态: 失败",
                            "执行方式: 多子句顺序执行",
                            f"关键结果: {last_summary or '执行失败'}",
                            f"总耗时: {cmd_cost:.2f} 秒",
                        ],
                        ok=False,
                    )
            except KeyboardInterrupt:
                self._stop_signal = True
                self.logger.info("[WORKER] 收到 KeyboardInterrupt，退出")
                break
            except Exception as e:
                self.logger.error(f"[WORKER] 任务执行异常: {e}\n{traceback.format_exc()}")
            finally:
                self._task_queue.task_done()
                # 简洁输出：在每个任务结束时（无论成功/失败），将关键结果推送到输出窗口
                try:
                    if getattr(self, 'output_window', None) is not None:
                        short = last_summary or ("执行完成" if overall_ok else "执行失败")
                        self.output_window.show_result(short)
                except Exception:
                    pass

    def Main_Control(self):
        """启动双线程智能体控制模式：看门狗(输入监听) + Agent Worker(顺序执行)。"""
        self.Agents_UAV()

    def save_detection_image(self, use_latest=False):
        """
        保存检测图。
        默认实时检测并保存；use_latest=True时使用最近检测缓存。
        """
        if self.save_detection_image_function is None:
            print("执行失败：未注入save_detection_image功能")
            return None
        return self.save_detection_image_function(use_latest=use_latest)

    def save_latest_detection_image(self):
        """
        保存最近一次检测缓存图（缓存不存在时会自动触发一次检测）。
        """
        if self.save_detection_image_function is None:
            print("执行失败：未注入save_detection_image功能")
            return None
        return self.save_detection_image_function(use_latest=True)

# ChatGpt交互模式控制类
'''
注意：本代码采用无人机的NED坐标系，室内动捕系统环境下飞行时，定义N向为动捕系统的X轴正方向，地面为高度0，向上为负
'''

import os
import ast
import time
import openai
import numpy as np
import cv2
import re
import traceback
import math
import sys
import types

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
        self.logger.info("OpenAI_APIs 初始化完成")

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

        print(f">>> {title}")
        print(header)
        for ln in safe_lines:
            print(ln)
        print(footer)

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

    def _handle_hard_rules(self, task: str):
        """
        极薄硬规则层：处理退出/急停与基础位移，其他语义交给LLM。
        返回(action, summary)
        action取值："continue" 表示已处理并进入下一轮；"pass" 表示交给LLM。
        """
        text = (task or "").strip()
        if not text:
            return "continue", "空指令"

        if text.lower() in self.ExitList:
            print("对话结束，程序退出。")
            self.logger.info("用户主动退出")
            raise KeyboardInterrupt

        # 急停硬规则：立即清零速度，避免模型生成延迟造成风险
        if re.search(r"急停|紧急停止|立即停止|stop\b", text, flags=re.IGNORECASE):
            try:
                self.MavList[0].SendVelFRD(0, 0, 0, 0)
                self.logger.warning("触发硬规则: 紧急停止")
                print("已执行紧急停止。")
                return "continue", "紧急停止已执行"
            except Exception as e:
                self.logger.error(f"紧急停止执行失败: {e}")
                print(f"紧急停止执行失败: {e}")
                return "continue", "紧急停止失败"

        # 基础位移硬规则：前后左右上下 + 米数，直接走确定性模板，避免LLM坐标映射偏差。
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
        """归一化常见目标别名，减少LLM在英文/中文目标名上的漂移。"""
        if not isinstance(name, str):
            return name
        text = name.strip()
        alias_map = {
            "drone": "uav",
            "无人机": "uav",
            "uav": "uav",
            "balloon": "balloon",
            "气球": "balloon",
            "red balloon": "red balloon",
            "blue ball": "blue ball",
            "小球": "blue ball",
        }
        return alias_map.get(text.lower(), alias_map.get(text, text))

    def _split_task_clauses(self, task: str):
        """
        将一条指令切分为多个子句，便于对子句进行模板拦截。
        """
        if not task:
            return []
        # 先按连接词切，再按常见标点切
        clauses = re.split(r"(?:然后|再|并且|并|,|，|;|；|。)", task)
        return [c.strip() for c in clauses if c and c.strip()]

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

        # 提取位移距离
        dist_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*米", text)
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
        # 允许“向左转45度 / 向左转向45度 / 右转45度”等口语变体
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
        m = re.search(r"(?:靠近|接近|飞向)(.+?)$", text)
        if not m:
            return None

        object_name = (m.group(1) or "").strip()
        object_name = re.sub(r"^(?:一下|下|一个|目标|物体)", "", object_name).strip()
        object_name = re.sub(r"(?:目标|物体|并停下|再停下|后停下)$", "", object_name).strip()
        if not object_name:
            return None
        return {"type": "approach", "object_name": object_name}

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

        object_name = (m.group(1) or "").strip()
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

    def _execute_approach_template(self, object_name: str):
        """
        执行靠近模板：搜索并持续逼近目标，直到满足停止条件。
        """
        if self.approachObjective_function is None:
            print("执行失败：未注入approach_objective功能")
            self.logger.warning("靠近模板拒绝执行: 未注入approachObjective_function")
            return False

        try:
            ok = self.approachObjective_function(object_name)
            summary = self._get_latest_result_cn(default_text="靠近完成")
            print(summary)
            self.logger.info(f"template_approach target={object_name} ok={bool(ok)} summary={summary}")
            return bool(ok)
        except Exception as e:
            print(f"执行失败：靠近模板异常 {e}")
            self.logger.error(f"靠近模板执行失败: {e}")
            self.logger.debug(traceback.format_exc())
            return False

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
        """
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
        ok = self.face_objective_function(object_name)
        if not ok:
            self.logger.warning(f"原地朝向目标失败: {object_name}")
            return False
        return True

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
                "deadline": time.monotonic() + 45.0,
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
        # 定义全局命名空间，包含当前类实例、time模块、body_to_ned函数和final_answer函数
        exec_globals = {
            "self": self,
            "time": time,
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
        # 为LLM生成代码注入受限工具包装器，避免参数污染与无限搜索循环。
        orig_search = self.search_object_function
        orig_approach = self.approachObjective_function
        orig_detect = self.detect_function
        orig_face = self.face_objective_function
        orig_strike = self.strike_objective_function

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
            self.logger.info("开始执行生成代码")
            self.logger.debug(f"代码内容:\n{clean_code}")
            # 执行代码
            exec(clean_code, exec_globals)
            self.logger.info("生成代码执行完成")
            return True
        except Exception as e:
            # 捕获并打印执行过程中可能出现的异常
            print(f"执行失败：{e}")
            self.logger.error(f"生成代码执行失败: {e}")
            self.logger.debug(traceback.format_exc())
            return False
        finally:
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

        # 主循环
        while True:
            try:
                # 获取用户输入的指令
                task = input("\n请输入你的控制模式指令: ").strip()
                # 如果用户输入exit或quit，退出程序
                if task.lower() in self.ExitList:
                    print("对话结束，程序退出。")
                    self.logger.info("用户主动退出")
                    break
                # 如果用户输入为空，提示重新输入
                if not task:
                    print("指令不能为空，请重新输入！")
                    continue
                self.logger.info(f"接收指令: {task}")

                clauses = self._split_task_clauses(task)
                if not clauses:
                    clauses = [task]

                cmd_start_time = time.time()
                cmd_id = datetime.now().strftime("%H%M%S")
                overall_ok = True
                last_summary = ""

                for idx, clause in enumerate(clauses, start=1):
                    self.logger.info(f"CLAUSE_START cmd_id={cmd_id} idx={idx}/{len(clauses)} clause={clause}")

                    action, summary = self._handle_hard_rules(clause)
                    if action == "continue":
                        # 硬规则执行后按语义判断该步是否成功。
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
                # 捕获键盘中断，退出程序
                print("\n检测到中断，程序退出。")
                self.logger.info("收到KeyboardInterrupt，退出主循环")
                break

    # 主控制逻辑
    def Main_Control(self):
        # 启动智能体模式
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

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
                 search_object_function, save_detection_image_function=None, face_objective_function=None):
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

        self.logger.info(f"template_face_object target={object_name}")
        ok = self.face_objective_function(object_name)
        if not ok:
            self.logger.warning(f"原地朝向目标失败: {object_name}")
            return False
        return True

    def _run_agent_for_clause(self, agent, clause: str):
        """
        执行单个非模板子句（走原 SmolAgents 流程）。
        """
        start_time = time.time()
        self.logger.info(f"开始请求模型生成代码, task={clause}")
        stream_steps = agent.run(clause, stream=True, max_steps=1)
        code = ""
        print("> > > > > " * 10)
        for step in stream_steps:
            if hasattr(step, "code_action") and step.code_action:
                code = step.code_action
                print(code)
        print("< < < < < " * 10)
        self.logger.info("模型代码生成完成")
        print("AI计算时间：", time.time() - start_time, "s")
        self.logger.info(f"AI计算时间: {time.time() - start_time:.3f}s")

        if code.strip():
            self.execute_generated_code(code)
            return True
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
        exec_globals = {"self": self, "time": time, "b2n": b2n, "final_answer": lambda x: print(f"执行成功：{x}")}
        # 兼容<code>和```python两种代码包裹格式
        code = code.strip()
        code_tag_match = re.search(r"<code>\s*([\s\S]*?)\s*</code>", code, flags=re.IGNORECASE)
        if code_tag_match:
            code = code_tag_match.group(1).strip()
        fence_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", code, flags=re.IGNORECASE)
        clean_code = fence_match.group(1).strip() if fence_match else code
        try:
            self.logger.info("开始执行生成代码")
            self.logger.debug(f"代码内容:\n{clean_code}")
            # 执行代码
            exec(clean_code, exec_globals)
            self.logger.info("生成代码执行完成")
        except Exception as e:
            # 捕获并打印执行过程中可能出现的异常
            print(f"执行失败：{e}")
            self.logger.error(f"生成代码执行失败: {e}")
            self.logger.debug(traceback.format_exc())

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

                # 子句级拦截：位移模板优先，其他子句走原 SmolAgents
                clauses = self._split_task_clauses(task)
                if not clauses:
                    print("指令解析失败，请重新输入。")
                    continue

                for idx, clause in enumerate(clauses, start=1):
                    self.logger.info(f"处理子句[{idx}/{len(clauses)}]: {clause}")
                    move_parsed = self._parse_body_move_clause(clause)
                    if move_parsed is not None:
                        if move_parsed.get("type") == "move_invalid":
                            print(f"执行失败：{move_parsed.get('reason', '方向冲突')}")
                            self.logger.warning(f"模板子句解析失败: {move_parsed}")
                            break
                        ok = self._execute_body_move_template(
                            dx_body=move_parsed["dx_body"],
                            dy_body=move_parsed["dy_body"],
                            dz_body=move_parsed["dz_body"],
                            distance_m=move_parsed["distance_m"],
                            direction_text=move_parsed["direction_text"],
                        )
                        if not ok:
                            print("子句执行失败，已中止后续子句执行。")
                            self.logger.warning("模板子句执行失败，终止本轮后续子句")
                            break
                        continue

                    turn_parsed = self._parse_turn_clause(clause)
                    if turn_parsed is not None:
                        ok = self._execute_turn_template(
                            sign=turn_parsed["sign"],
                            deg=turn_parsed["deg"],
                            turn_text=turn_parsed["turn_text"],
                        )
                        if not ok:
                            print("子句执行失败，已中止后续子句执行。")
                            self.logger.warning("模板子句执行失败，终止本轮后续子句")
                            break
                        continue

                    face_parsed = self._parse_face_object_clause(clause)
                    if face_parsed is not None:
                        ok = self._execute_face_object_template(face_parsed["object_name"])
                        if not ok:
                            print("子句执行失败，已中止后续子句执行。")
                            self.logger.warning("朝向目标子句执行失败，终止本轮后续子句")
                            break
                        continue

                    ok = self._run_agent_for_clause(agent, clause)
                    if not ok:
                        print("子句执行失败，已中止后续子句执行。")
                        self.logger.warning("智能体子句执行失败，终止本轮后续子句")
                        break
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

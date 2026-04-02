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
                 search_object_function, save_detection_image_function=None):
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
                # 记录开始时间
                start_time = time.time()
                self.logger.info(f"接收指令: {task}")
                self.logger.info(f"开始请求模型生成代码, task={task}")
                # 运行智能体，获取流式响应
                stream_steps = agent.run(task, stream=True, max_steps=1)
                code = ""
                print("> > > > > " * 10)
                for step in stream_steps:
                    if hasattr(step, "code_action") and step.code_action:
                        code = step.code_action
                        print(code)
                print("< < < < < " * 10)
                self.logger.info("模型代码生成完成")
                # 打印AI计算时间
                print("AI计算时间：", time.time() - start_time, "s")
                self.logger.info(f"AI计算时间: {time.time() - start_time:.3f}s")

                # 执行生成的代码
                if code.strip():
                    self.execute_generated_code(code)
                else:
                    self.logger.warning("本轮未收到可执行代码")
                    print("未生成可执行代码，请重试指令。")
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

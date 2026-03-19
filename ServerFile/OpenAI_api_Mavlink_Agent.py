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

from datetime import datetime, timezone
from Description import Description as Des
from Coordinate_Transformation import body_to_ned as b2n

from smolagents import CodeAgent, PromptTemplates, PlanningPromptTemplate, ManagedAgentPromptTemplate, \
    FinalAnswerPromptTemplate
from volcEngineLLM import VolcEngineFakeHFModel


class OpenAI_APIs(Des):
    version = "3.2"

    def __init__(self, MavList, VehilceNum, detect_function, approachObjective_function, look_function,
                 search_object_function):
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

        # 设置火山引擎API密钥
        os.environ['OPENAI_API_KEY'] = "24572520-5c64-4470-8c3d-5ecb84781725"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # 设置火山引擎API的基础URL
        self.client = openai.OpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3 ")
        # 设置使用的语言模型
        self.LLMModel = "deepseek-v3-250324"
        # 初始化聊天历史记录
        self.chatHistory = []

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
        # 使用正则表达式清理代码，去除多余的Markdown格式
        clean_code = re.sub(r"^```python\s*|\s*```$", "", code.strip())
        try:
            # 执行代码
            exec(clean_code, exec_globals)
        except Exception as e:
            # 捕获并打印执行过程中可能出现的异常
            print(f"执行失败：{e}")

    # 智能体模式
    def Agents_UAV(self):
        # 定义提示模板
        prompt_templates = PromptTemplates(
            system_prompt=self.Prompt_dit["Prompt_smol"],  # 系统提示
            planning=PlanningPromptTemplate(  # 规划提示
                initial_plan="",
                update_plan_pre_messages="",
                update_plan_post_messages="",
            ),
            managed_agent=ManagedAgentPromptTemplate(  # 管理代理提示
                task="",
                report="",
            ),
            final_answer=FinalAnswerPromptTemplate(  # 最终答案提示
                pre_messages="",
                post_messages="",
            ),
        )

        # 初始化CodeAgent，指定模型和提示模板
        agent = CodeAgent(model="deepseek-v3", prompt_templates=prompt_templates, tools=[])
        # 将模型替换为VolcEngineFakeHFModel
        agent.model = VolcEngineFakeHFModel()

        # 主循环
        while True:
            try:
                # 获取用户输入的指令
                task = input("\n请输入你的控制模式指令: ").strip()
                # 如果用户输入exit或quit，退出程序
                if task.lower() in self.ExitList:
                    print("对话结束，程序退出。")
                    break
                # 如果用户输入为空，提示重新输入
                if not task:
                    print("指令不能为空，请重新输入！")
                    continue
                # 记录开始时间
                start_time = time.time()
                # 运行智能体，获取流式响应
                stream_steps = agent.run(task, stream=True, max_steps=1)
                print("> > > > > " * 10)
                for step in stream_steps:
                    # 如果有代码动作，提取代码并打印
                    if hasattr(step, "code_action") and step.code_action:
                        code = step.code_action
                        print(code)
                print("< < < < < " * 10)
                # 打印AI计算时间
                print("AI计算时间：", time.time() - start_time, "s")

                # 执行生成的代码
                self.execute_generated_code(code)
            except KeyboardInterrupt:
                # 捕获键盘中断，退出程序
                print("\n检测到中断，程序退出。")
                break

    # 主控制逻辑
    def Main_Control(self):
        # 启动智能体模式
        self.Agents_UAV()

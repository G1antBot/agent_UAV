**实验6-7无人机视觉语言控制软件在环仿真实验**

1.  **实验背景**

无人机技术的迅猛发展使其在军事、民用和科研领域的重要性日益凸显。飞行控制系统的可靠性、安全性和性能是决定无人机任务成败的关键因素。为了降低开发风险、缩短迭代周期，软件在环（Software-in-the-Loop,
SIL）仿真技术已成为验证和优化控制算法的重要手段。通过将飞行控制软件嵌入高保真虚拟环境，SIL仿真技术能够在没有实物平台的情况下完成算法测试，显著降低试验成本并提高研发效率。

SIL仿真在无人机开发流程中的价值主要体现在以下三个方面：首先，在设计初期，它为控制算法提供了闭环验证平台，能够提前发现潜在缺陷，从而避免实飞事故的发生；其次，它能够复现极端或复杂的场景（如多变气象条件、崎岖地形、特定任务剖面等），从而全面评估系统的鲁棒性和适应性；最后，它支持参数在线调优，实现飞行效率与稳定性的同步提升。

本实验旨在利用软件在环仿真技术深入研究无人机的飞行控制策略。我们将使用飞思集群仿真平台和MATLAB等工具构建一个虚拟飞行环境，模拟无人机的飞行过程。实验中，我们将通过Python大语言模型自动代码生成对无人机进行控制，实现无人机通过指令识别不同颜色的小球并靠近。通过这种方式，我们不仅可以验证控制算法的有效性，还可以对无人机的飞行性能进行评估和优化。

在无人机技术快速发展的当下，软件在环仿真技术为无人机控制系统的研究和开发提供了一种高效、安全且经济的方法。通过本实验，我们希望深入理解单机控制的基本概念，掌握通过Python语言控制无人机的方法，并学会使用飞思集群仿真平台完成单机软件在环仿真。这将为后续的无人机控制系统开发和优化奠定坚实的基础，推动无人机技术在各个领域的广泛应用和发展。

2.  **实验目的**

<!-- -->

1.  理解单机控制的基本概念，熟悉单机接口配置。

2.  掌握通过Python大语言模型自动代码生成控制指令使无人机识别不同颜色的小球并靠近。

3.  学会使用飞思集群仿真平台、Python完成单机软件在环仿真。

<!-- -->

3.  **实验原理与说明**

### **理论基础**

本实验通过Mavlink通信协议、开放词汇目标检测（Grounding
DINO）、代码化智能体（SmolAgents）与火山引擎大语言模型（LLM）的深度融合，实现了"自然语言指令→大模型决策→无人机控制→环境感知"的闭环仿真。实验核心是将人类的自然语言需求转化为无人机可执行的控制逻辑，本质是跨模态感知（视觉-文本）+代码化决策+模型预测控制的综合应用。以下从系统架构、关键模块原理、闭环逻辑三个层面展开说明，并结合公式与数学推导深化理解。

#### **3.1.1系统整体架构**

实验系统遵循"感知-决策-控制"的经典机器人闭环框架，各模块通过数据流动形成完整链路：

graph TD

A\[自然语言指令\] \--\> B\[SmolAgents大模型\]

B \--\> C\[生成Python控制代码\]

C \--\> D\[执行代码：调用感知/控制接口\]

D \--\> E\[Mavlink通信：获取无人机状态\]

E \--\> F\[Grounding DINO：目标检测\]

F \--\> G\[LLM图像理解：提取目标列表\]

G \--\> B\[决策层：更新控制策略\]

D \--\> H\[PX4MavCtrl：发送控制指令\]

H \--\> I\[无人机：执行动作（旋转、逼近）\]

I \--\> E\[感知层：反馈新状态\]

- 感知层：通过Mavlink获取无人机状态（位置、姿态），通过Grounding
  DINO实现开放词汇目标检测，通过LLM理解图像内容；

- 决策层：通过SmolAgents将自然语言指令转化为Python代码，调用感知层结果生成控制策略；

- 控制层：通过PX4MavCtrl发送NED坐标系下的位置/速度指令，实现无人机的姿态调整与目标逼近。

#### **3.1.2关键模块原理与数学推导**

1.  **Mavlink通信与坐标系转换：无人机状态感知基础**

Mavlink（Micro Air Vehicle
Link）是无人机领域的轻量级通信协议，负责无人机与地面站/仿真环境之间的状态传输（如位置、姿态、速度）和指令下发（如位置控制、姿态调整）。实验中BodyCommMavlink类封装了Mavlink的初始化与状态获取逻辑，核心是坐标系转换与状态同步。

1.  **坐标系定义与偏移计算**

无人机的运动控制基于NED坐标系（North-East-Down，北-东-地），而UE4仿真环境的全局坐标系为ENU-like（X-东、Y-北、Z-上，与NED存在旋转变换）。为了统一两个坐标系的位置表示，实验通过偏移量将NED坐标转换为UE4全局坐标：

对于第i架无人机，设其NED坐标系下的位置为uavPosNED = \[Nx, Ny,
Nz\]（北、东、地），UE4全局坐标系下的位置为uavGlobalPos = \[Gx, Gy,
Gz\]（东、北、上）。由于NED的"地"轴（Z）与UE4的"上"轴（Z）方向相反，且X/Y轴顺序不同，需计算坐标系偏移量：

![](media/image1.wmf)

其中负号用于抵消NED与UE4的轴方向差异。转换后的UE4全局坐标为：

![](media/image2.wmf)

2.  **Mavlink状态同步流程**

BodyCommMavlink类通过InitMavLoop初始化Mavlink通信线程，实时获取无人机的：

- 位置：uavPosNED（NED坐标系）；

- 姿态：uavAngEular（欧拉角：滚转、俯仰、偏航，单位：弧度）；

- 速度：uavVelNED（NED坐标系下的线速度）。

> 这些状态是后续控制算法的输入源，确保决策层能基于无人机的实时状态调整策略。

2.  **开放词汇目标检测：Grounding DINO 的跨模态感知**

实验中detect_dino函数实现了开放词汇目标检测（即无需预训练类别，直接通过文本指令检测目标），核心是Grounding
DINO模型------一种结合视觉与文本的跨模态Transformer模型。

1.  **Grounding DINO 原理概述**

Grounding
DINO的创新点是统一的跨模态融合框架，将视觉特征（图像）与文本特征（目标名称）编码后，通过Transformer解码器进行关联，最终输出目标的边界框与置信度。其结构包括：

- 视觉编码器：Swin Transformer，将图像转换为视觉特征图；

- 文本编码器：BERT，将目标名称（如"red ball"）转换为文本特征序列；

- 跨模态解码器：Transformer解码器，融合视觉与文本特征，预测目标的边界框与分类置信度。

2.  **图像预处理与特征提取**

实验中图像预处理流程遵循Grounding DINO的要求：

![](media/image3.wmf)

其中：

- RandomResize：将图像短边缩至800，长边不超过1333（保持aspect ratio）；

- ToTensor：将图像从H×W×C（OpenCV格式）转换为C×H×W（PyTorch格式）；

- Normalize：用ImageNet的均值\[0.485, 0.456, 0.406\]和标准差\[0.229,
  0.224, 0.225\]归一化，消除图像亮度差异。

3.  **目标检测与输出**

Grounding DINO的预测过程可表示为：

![](media/image4.wmf)

其中：

- TextPrompt：输入的目标名称（如"airplane"）；

- BoxThreshold（0.25）：边界框置信度阈值，低于此值的框被过滤；

- TextThreshold（0.15）：文本-视觉匹配置信度阈值，低于此值的目标被过滤；

- Boxes：目标边界框（格式为cxcywh，即中心坐标+宽高），需转换为xyxy（左上角+右下角）：

> ![](media/image5.wmf)

4.  **损失函数（模型训练层面）**

Grounding DINO的训练损失由三部分组成：

![](media/image6.wmf)

- 分类损失（L_cls）：交叉熵损失，计算预测类别与真实类别的差异；

- 边界框损失（L_box）：GIoU损失，衡量预测框与真实框的重叠度；

- 文本-视觉匹配损失（L_text）：对比损失，最大化目标文本与视觉特征的相似度，最小化非目标文本的相似度。

3.  **大模型决策：SmolAgents与代码化智能体**

实验中Agents_UAV函数实现了自然语言到代码的转换，核心是SmolAgents框架------一种基于代码的智能体，能将自然语言指令转化为可执行的Python代码，并调用无人机控制接口。

1.  **SmolAgents原理**

SmolAgents的核心是"提示工程+代码生成"，通过定义系统提示（Prompt_smol）引导大模型生成符合要求的代码。其流程为：

- 输入解析：将用户的自然语言指令（如"找到红色气球并靠近"）转换为任务描述；

- 代码生成：大模型（火山引擎deepseek-v3）根据系统提示生成Python代码；

- 代码执行：execute_generated_code函数去除代码块标记（\`\`\`python），执行代码；

- 结果反馈：将执行结果（如"目标找到"）返回给大模型，调整后续策略。

2.  **火山引擎LLM的调用**

实验中VolcEngineFakeHFModel类将火山引擎的API包装为Hugging
Face（HF）模型格式，使SmolAgents能直接调用。其核心是请求-响应流程：

![](media/image7.wmf)

其中：

- ApiUrl：火山引擎LLM接口（https://ark.cn-beijing.volces.com/api/v3/chat/completions）；

- Headers：包含API密钥的授权信息；

- Payload：包含模型名称（deepseek-v3-250324）、对话历史、温度（temperature=0.7）等参数。

#### **3.1.3闭环逻辑与实验流程**

实验的闭环逻辑是"感知-决策-控制-反馈"的循环，以"找到红色气球并靠近"为例，流程如下：

1.  **输入指令**

> 用户输入自然语言指令："找到红色气球并靠近"。

2.  **大模型决策**

> SmolAgents引导火山引擎LLM生成代码：

\# 搜索红色气球

found = self.search_object(\"red ball\")

if found:

\# 检测目标位置

obj_list, obj_locs, \_, \_ = self.detect_dino(\"red ball\")

if \"red ball\" in obj_list:

\# 获取目标中心误差

target_x, target_y = obj_locs\[0\]\[0\] +
(obj_locs\[0\]\[2\]-obj_locs\[0\]\[0\])/2, obj_locs\[0\]\[1\] +
(obj_locs\[0\]\[3\]-obj_locs\[0\]\[1\])/2

ex = self.vis.Img\[0\].shape\[1\]/2 - target_x

ey = self.vis.Img\[0\].shape\[0\]/2 - target_y

\# 逼近目标

self.approachObjective(ex, ey)

3.  **代码执行与控制**

- 搜索目标：search_object函数控制无人机旋转，每次转40°，检查目标是否存在；

- 检测位置：detect_dino返回目标的边界框，计算目标中心在图像中的误差ex（x轴）与ey（y轴）；

- 逼近目标：approachObjective先调整偏航（YAW_ALIGN），再分解速度（APPROACH），控制无人机向目标移动。

4.  **状态反馈与调整**

无人机执行控制指令后，Mavlink实时反馈新的位置与姿态，detect_dino重新检测目标位置，调整ex与ey，形成闭环控制，直到目标进入图像中心（\|ex\|≤1且\|ey\|≤1）。

#### **3.1.4实验意义和结论**

本实验的创新点是将大模型的代码生成能力与无人机的感知控制结合，实现了"自然语言-代码-控制"的端到端闭环。其核心原理包括：

- Mavlink通信：实现无人机状态的实时获取与指令下发；

- Grounding DINO：实现开放词汇的跨模态目标检测；

- SmolAgents：将自然语言转化为可执行代码；

- 阶段式控制：实现精准的偏航对准与目标逼近。

通过这些模块的协同，实验成功将人类的自然语言需求转化为无人机的自主行为，为无人机的智能化控制提供了新的思路。未来可通过优化大模型的提示工程、增加PID控制等方式，进一步提升控制精度与鲁棒性。

### **源码分析**

#### **3.2.1 volcEngineLLM.py代码解析**

用Visual Studio
Code软件打开[1.软件在环实验\\ServerFile\\volcEngineLLM.py](1.软件在环实验/ServerFile/volcEngineLLM.py)文件，这段代码定义了一个名为VolcEngineFakeHFModel的类，用于模拟与火山引擎语言模型API的交互。它初始化API密钥、地址和模型ID，通过generate方法接收消息列表，序列化后发送POST请求到火山引擎API。成功时，解析响应并提取代码块内容，返回模拟的消息对象；失败时，捕获异常并返回失败消息。

volcEngineLLM.py代码解析：

import requests

import re

class VolcEngineFakeHFModel:

def \_\_init\_\_(self):

\# 初始化API密钥、API地址和模型ID

self.api_key = \"24572520-5c64-4470-8c3d-5ecb84781725\" \#
火山引擎API密钥

self.api_url =
\"https://ark.cn-beijing.volces.com/api/v3/chat/completions\" \#
火山引擎API地址

self.model_id = \"deepseek-v3-250324\" \# 使用的模型ID

def generate(self, messages, \*\*kwargs):

\# 将传入的消息对象序列化为字典格式，因为API需要这种格式

serialized_messages = \[{\"role\": m.role, \"content\": m.content} for m
in messages\]

\# 设置请求头，包括内容类型和授权信息

headers = {\"Content-Type\": \"application/json\", \"Authorization\":
f\"Bearer {self.api_key}\"}

\# 构造请求负载，包括模型ID、消息序列、温度参数和是否流式传输

payload = {\"model\": self.model_id, \"messages\": serialized_messages,
\"temperature\": 0.7, \"stream\": False}

\# 定义一个模拟的Token使用情况类，用于返回模拟的token使用信息

class FakeTokenUsage:

def \_\_init\_\_(self):

self.input_tokens = 0 \# 输入token数量

self.output_tokens = 0 \# 输出token数量

self.total_tokens = 0 \# 总token数量

try:

\# 发送POST请求到API

response = requests.post(self.api_url, headers=headers, json=payload)

response.encoding = \'utf-8\' \# 设置响应编码为UTF-8

response.raise_for_status() \# 如果响应状态码不是200，抛出异常

\# 解析响应内容，获取生成的消息内容

raw_content =
response.json()\[\"choices\"\]\[0\]\[\"message\"\]\[\"content\"\]

\# 使用正则表达式提取代码块内容，去除其他说明性文字

code_match = re.search(r\"\<code\>(.\*?)\</code\>\", raw_content,
re.DOTALL)

if code_match:

code = code_match.group(1).strip() \# 如果找到代码块，提取并去除首尾空格

else:

code = raw_content.strip() \#
如果没有代码块，直接使用原始内容并去除首尾空格

\# 定义一个模拟的消息类，用于返回处理后的消息内容

class FakeMessage:

def \_\_init\_\_(self, content):

self.content = f\"\<code\>\\n{code}\\n\</code\>\" \#
将代码内容包装在\<code\>标签中

self.token_usage = FakeTokenUsage() \# 初始化模拟的token使用情况

print(f\"火山方舟 API 调用成功。\") \# 打印成功信息

return FakeMessage(code) \# 返回模拟的消息对象

except Exception as e:

\# 如果API调用失败，打印错误信息

print(f\"火山方舟 API 调用失败: {e}\")

\# 定义一个模拟的消息类，用于返回失败的响应

class FakeMessage:

def \_\_init\_\_(self, content):

self.content = (\"\<code\>\\nprint(\'失败\')\\n\</code\>\") \#
返回一个简单的失败代码

self.token_usage = FakeTokenUsage() \# 初始化模拟的token使用情况

return FakeMessage(\"【模型响应失败】\") \# 返回模拟的消息对象

#### **3.2.2** **OpenAI_api_Mavlink_Agent.py代码解析**

用Visual Studio
Code软件打开[1.软件在环实验\\ServerFile\\OpenAI_api_Mavlink_Agent.py](1.软件在环实验/ServerFile/OpenAI_api_Mavlink_Agent.py)文件，这段代码定义了一个名为OpenAI_APIs的类，用于实现与火山引擎LLM的交互，控制无人机。它初始化无人机列表、数量及功能函数，配置API密钥和模型。类中包含记录聊天历史、执行生成代码的方法，以及智能体模式，允许用户输入指令，模型生成并执行代码，记录AI计算时间。主控制逻辑启动智能体模式，程序结束后打印聊天历史。

OpenAI_api_Mavlink_Agent.py代码解析：

\# ChatGpt交互模式控制类

\'\'\'

注意：本代码采用无人机的NED坐标系，室内动捕系统环境下飞行时，定义N向为动捕系统的X轴正方向，地面为高度0，向上为负

\'\'\'

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

from smolagents import CodeAgent, PromptTemplates,
PlanningPromptTemplate, ManagedAgentPromptTemplate,
FinalAnswerPromptTemplate

from volcEngineLLM import VolcEngineFakeHFModel

class OpenAI_APIs(Des):

version = \"3.2\"

def \_\_init\_\_(self, MavList, VehilceNum, detect_function,
approachObjective_function, look_function, search_object_function):

\# 调用父类的初始化方法

super().\_\_init\_\_()

\# 初始化无人机列表和数量

self.MavList = MavList

self.VehilceNum = VehilceNum

\# 初始化功能函数，用于无人机的特定操作

self.detect_function = detect_function

self.approachObjective_function = approachObjective_function

self.look_function = look_function

self.search_object_function = search_object_function

\# 设置火山引擎API密钥

os.environ\[\'OPENAI_API_KEY\'\] =
\"24572520-5c64-4470-8c3d-5ecb84781725\"

openai.api_key = os.getenv(\"OPENAI_API_KEY\")

\# 设置火山引擎API的基础URL

self.client = openai.OpenAI(base_url =
\"https://ark.cn-beijing.volces.com/api/v3 \")

\# 设置使用的语言模型

self.LLMModel = \"deepseek-v3-250324\"

\# 初始化聊天历史记录

self.chatHistory = \[\]

\# 记录聊天历史记录

def GetHistrory(self, prompt, Answer):

\# 获取当前UTC时间

UTCTime = datetime.now(timezone.utc)

\# 格式化时间

TimeTemp = UTCTime.strftime(\"%Y-%m-%d %H:%M:%S %Z\")

\# 将时间、问题和回答记录到聊天历史中

self.chatHistory.append({\"Time\": TimeTemp, \"Qustion\": prompt,
\"Answer\": Answer})

\# 执行生成的代码

def execute_generated_code(self, code: str):

\#
定义全局命名空间，包含当前类实例、time模块、body_to_ned函数和final_answer函数

exec_globals = {\"self\": self, \"time\": time, \"b2n\": b2n,
\"final_answer\": lambda x : print(f\"执行成功：{x}\")}

\# 使用正则表达式清理代码，去除多余的Markdown格式

clean_code = re.sub(r\"\^\`\`\`python\\s\*\|\\s\*\`\`\`\$\", \"\",
code.strip())

try:

\# 执行代码

exec(clean_code, exec_globals)

except Exception as e:

\# 捕获并打印执行过程中可能出现的异常

print(f\"执行失败：{e}\")

\# 智能体模式

def Agents_UAV(self):

\# 定义提示模板

prompt_templates = PromptTemplates(

system_prompt=self.Prompt_dit\[\"Prompt_smol\"\], \# 系统提示

planning=PlanningPromptTemplate( \# 规划提示

initial_plan=\"\",

update_plan_pre_messages=\"\",

update_plan_post_messages=\"\",

),

managed_agent=ManagedAgentPromptTemplate( \# 管理代理提示

task=\"\",

report=\"\",

),

final_answer=FinalAnswerPromptTemplate( \# 最终答案提示

pre_messages=\"\",

post_messages=\"\",

),

)

\# 初始化CodeAgent，指定模型和提示模板

agent = CodeAgent(model=\"deepseek-v3\",
prompt_templates=prompt_templates, tools=\[\])

\# 将模型替换为VolcEngineFakeHFModel

agent.model = VolcEngineFakeHFModel()

\# 主循环

while True:

try:

\# 获取用户输入的指令

task = input(\"\\n请输入你的控制模式指令: \").strip()

\# 如果用户输入exit或quit，退出程序

if task.lower() in self.ExitList:

print(\"对话结束，程序退出。\")

break

\# 如果用户输入为空，提示重新输入

if not task:

print(\"指令不能为空，请重新输入！\")

continue

\# 记录开始时间

start_time = time.time()

\# 运行智能体，获取流式响应

stream_steps = agent.run(task, stream=True, max_steps=1)

print(\"\> \> \> \> \> \" \* 10)

for step in stream_steps:

\# 如果有代码动作，提取代码并打印

if hasattr(step, \"code_action\") and step.code_action:

code = step.code_action

print(code)

print(\"\< \< \< \< \< \" \* 10)

\# 打印AI计算时间

print(\"AI计算时间：\", time.time() - start_time, \"s\")

\# 执行生成的代码

self.execute_generated_code(code)

except KeyboardInterrupt:

\# 捕获键盘中断，退出程序

print(\"\\n检测到中断，程序退出。\")

break

\# 主控制逻辑

def Main_Control(self):

\# 启动智能体模式

self.Agents_UAV()

#### **3.2.3 Communication_Mavlink.py代码解析**

用Visual Studio
Code软件打开[1.软件在环实验\\ServerFile\\Communication_Mavlink.py](1.软件在环实验/ServerFile/Communication_Mavlink.py)文件，这段代码定义了一个名为BodyCommMavlink的类，用于实现无人机的Mavlink通信、目标检测和跟踪功能。它初始化无人机的Mavlink连接和前置摄像头，加载目标检测模型（Grounding
DINO），并提供方法进行目标检测、搜索目标、图像处理和接近目标。通过与火山引擎LLM的交互，它还支持图像理解，返回图像中的主要目标名称。

Communication_Mavlink.py代码解析：

\# Mavlink通信文件

import cv2

import numpy as np

import time

import VisionCaptureApi

import PX4MavCtrlV4 as PX4MavCtrl

import ReqCopterSim

from openai import OpenAI

import base64

import torch

from PIL import Image

from torchvision.ops import box_convert

from groundingdino.util.inference import load_model, load_image,
predict, annotate

import groundingdino.datasets.transforms as T

from PIL import Image

import math

class BodyCommMavlink(object):

def \_\_init\_\_(self):

\# 检查是否使用GPU

if torch.cuda.is_available():

print(\"use_gpu\")

self.is_cup = False

else:

print(\"use_cpu\")

self.is_cup = True

\# 初始化火山引擎LLM客户端

api_key = \"24572520-5c64-4470-8c3d-5ecb84781725\"

self.llm_client = OpenAI(

api_key=api_key,

base_url=\"https://ark.cn-beijing.volces.com/api/v3 \",

)

\# 加载Grounding DINO模型

self.dino_model =
load_model(\"groundingdino/config/GroundingDINO_SwinT_OGC.py\",
\"groundingdino/groundingdino_swint_ogc.pth\")

self.BOX_TRESHOLD = 0.25 \# 目标检测的框阈值

self.TEXT_TRESHOLD = 0.15 \# 文本阈值

\# 初始化ReqCopterSim实例，用于与模拟器通信

self.req = ReqCopterSim.ReqCopterSim()

StartCopterID = 1 \# 起始无人机ID

TargetIP = self.req.getSimIpID(StartCopterID) \# 获取目标模拟器的IP地址

\# 初始化VisionCaptureApi实例，用于获取前置摄像头的图像

self.vis = VisionCaptureApi.VisionCaptureApi(TargetIP)

self.vis.jsonLoad() \# 加载配置文件

self.vis.sendReqToUE4(0, TargetIP) \# 向UE4发送请求

self.vis.startImgCap() \# 启动图像捕获

\# 初始化无人机列表

self.VehilceNum = 1 \# 无人机数量

self.MavList = \[\]

for i in range(self.VehilceNum):

CopterID = StartCopterID + i \# 当前无人机ID

TargetIP = self.req.getSimIpID(CopterID) \# 获取目标模拟器的IP地址

self.req.sendReSimIP(CopterID) \# 将无人机连接到指定的模拟器IP地址

time.sleep(1)

self.MavList = self.MavList + \[PX4MavCtrl.PX4MavCtrler(CopterID,
TargetIP)\] \# 创建无人机控制器实例并添加到列表中

time.sleep(2)

\# 初始化Mavlink循环

for i in range(self.VehilceNum):

self.MavList\[i\].InitMavLoop() \# 初始化每架无人机的Mavlink循环

time.sleep(2)

\# 计算全局坐标（UE4地图）与NED坐标（无人机本地）的偏移量

self.Error2UE4Map = \[\]

for i in range(self.VehilceNum):

mav = self.MavList\[i\]

self.Error2UE4Map = self.Error2UE4Map + \[

-np.array(\[

mav.uavGlobalPos\[0\] - mav.uavPosNED\[0\], \# X轴偏移

mav.uavGlobalPos\[1\] - mav.uavPosNED\[1\], \# Y轴偏移

mav.uavGlobalPos\[2\] - mav.uavPosNED\[2\] \# Z轴偏移

\])

\]

def GetBodyMavList(self):

\# 返回无人机列表、无人机数量和坐标偏移量

return self.MavList, self.VehilceNum, self.Error2UE4Map

def detect_dino(self, object_names):

\# 使用Grounding DINO模型进行目标检测

IMAGE_PATH = \".asset/cats.png\"

transform = T.Compose(

\[

T.RandomResize(\[800\], max_size=1333),

T.ToTensor(),

T.Normalize(\[0.485, 0.456, 0.406\], \[0.229, 0.224, 0.225\]),

\]

)

\# 获取前置摄像头的图像

image = self.vis.Img\[0\].copy()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

\# 将图像转换为PIL格式

image_source = Image.fromarray(image)

\# 应用图像变换

image_transformed, \_ = transform(image_source, None)

\# 目标检测

TEXT_PROMPT = object_names

boxes, logits, phrases = predict(

model=self.dino_model,

image=image_transformed,

caption=TEXT_PROMPT,

box_threshold=self.BOX_TRESHOLD,

text_threshold=self.TEXT_TRESHOLD,

device=torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")

)

\# 标注检测结果

annotated_frame = annotate(image_source=image, boxes=boxes,
logits=logits, phrases=phrases)

cv2.imwrite(\"annotated_image.jpg\", annotated_frame)

\# 返回检测结果

obj_list = phrases \# 使用预测的短语作为对象列表

h, w, \_ = image.shape

boxes = boxes \* torch.Tensor(\[w, h, w, h\])

boxes = box_convert(boxes=boxes, in_fmt=\"cxcywh\",
out_fmt=\"xyxy\").numpy()

obj_locs = boxes.tolist() \# 转换为Python列表

obj_logits = logits.tolist()

img_with_box = Image.fromarray(annotated_frame) \# 转换为PIL格式

return obj_list, obj_locs, obj_logits, img_with_box

def search_object(self, object_names):

\# 通过旋转无人机的偏航角搜索目标

current_yaw = self.MavList\[0\].uavAngEular\[2\]

for yaw_step in range(0, 360, 40):

new_yaw = current_yaw + (yaw_step \* 3.14159 / 180) \# 转换为弧度

self.MavList\[0\].SendPosNED(

self.MavList\[0\].uavPosNED\[0\],

self.MavList\[0\].uavPosNED\[1\],

self.MavList\[0\].uavPosNED\[2\],

new_yaw

)

time.sleep(2)

\# 检测目标

obj_list, obj_locs, obj_logits, img_with_box =
self.detect_dino(object_names)

if object_names in obj_list:

print(object_names, \" found during rotation.\")

return True

return False

def cv2_to_base64(self, image, format=\'.png\'):

\# 将OpenCV图像转换为Base64字符串

success, buffer = cv2.imencode(format, image)

if not success:

raise ValueError(\"图片编码失败，请检查格式参数\")

img_bytes = buffer.tobytes()

return base64.b64encode(img_bytes).decode(\'utf-8\')

def look(self):

\# 获取前置摄像头的图像，并通过火山引擎LLM进行图像理解

rgb_image = self.vis.Img\[0\]

base64_str = self.cv2_to_base64(rgb_image, \".png\")

response = self.llm_client.chat.completions.create(

model=\"doubao-1-5-vision-pro-32k-250115\",

messages=\[

{

\"role\": \"user\",

\"content\": \[

{\"type\": \"text\", \"text\":
\"图片中有哪些目标，请给出名称即可，给出常见的，清晰可见的目标即可，多个目标名称之间用英文逗号分隔\"},

{

\"type\": \"image_url\",

\"image_url\": {

\"url\": f\"data:image/png;base64,{base64_str}\"

}

},

\],

}

\],

temperature=0.01

)

content = response.choices\[0\].message.content

return content

def approachObjective(self, error_x, error_y):

\"\"\"

根据目标的误差控制无人机接近目标。

:param error_x: 目标在X方向上的误差（像素值）。

:param error_y: 目标在Y方向上的误差（像素值）。

\"\"\"

\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-- 一次性初始化
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

if not hasattr(self, \"\_servo\"):

\# 检查是否使用CPU

is_cpu = bool(getattr(self, \"is_cup\", True))

\# 根据是否使用CPU，设置检测帧率

det_fps = 0.4 if is_cpu else 10.0

self.\_servo = {

\# ------ 阶段控制 ------

\"phase\": \"YAW_ALIGN\", \# 初始阶段：先对准偏航

\"yaw_hold_need\": 3, \# 连续满足阈值的次数，用于确认偏航对准

\"yaw_hold_cnt\": 0, \# 当前连续满足阈值的计数

\# ------ 误差处理 ------

\"tau_err\": 0.5 if is_cpu else 0.25, \#
低通滤波器的时间常数，用于平滑误差

\"db_x\": 5.0, \"db_y\": 5.0, \# 死区（像素/角度），避免误差过小时的抖动

\"hit_x\": 1.0, \"hit_y\": 1.0, \# 到达阈值，用于判断是否到达目标

\"hit_need\": 3, \# 连续命中次数，用于确认到达目标

\# ------ 偏航控制 ------

\"K_yaw\": 0.0006, \# 偏航增益，控制偏航角速度的大小

\"yaw_max\": math.radians(30), \# 最大偏航角速度（弧度/秒）

\"yaw_align_tol\": 25.0, \# 认为"对准"的偏航误差阈值（像素/角度）

\# ------ 速度合成（朝向目标） ------

\# 先 yaw 对准；对准后速度指向 X--Z 平面内"朝向目标"的方向

\# 用 ey -\> alpha（俯仰方向角）来分解： vx=v\*cos(alpha),
vz=v\*sin(alpha)

\"ay\": 327.0, \# ey-\>alpha 的尺度，越小越敏感

\"alpha_max\": math.radians(85), \# 最大俯仰方向角，防止直冲上下

\"v_nom\": 0.5, \# 对准后朝向推进的标称速度

\"v_min\": 0.05, \"v_max\": 1.0, \# 推进速度标量上下限

\"vz_max\": 0.35, \# 垂直分量限幅（FRD：向下为正）

\# ------ 安全与下发 ------

\"lost_timeout\": max(3.0/det_fps, 1.5), \# 目标丢失超时时间

\"hold_sec\": max(0.8/det_fps, 0.15), \# 指令保持时间

\# ------ 运行态 ------

\"last_time\": time.monotonic(), \# 上次运行时间

\"last_det_ts\": time.monotonic(), \# 上次检测到目标的时间

\"lp_ex\": 0.0, \"lp_ey\": 0.0, \# 低通滤波后的误差

\"hit_cnt\": 0, \# 连续命中计数

\"last_cmd\": (0.0, 0.0, 0.0, 0.0), \# 上次发送的指令

\"next_ok_ts\": 0.0, \# 下次可以发送指令的时间

}

s = self.\_servo

t = time.monotonic() \# 当前时间

dt = t - s\[\"last_time\"\]; s\[\"last_time\"\] = t \#
计算时间差并更新上次运行时间

s\[\"last_det_ts\"\] = t \# 更新上次检测到目标的时间

\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-- 小工具
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

def clamp(v, vmin, vmax):

\"\"\"

限制值v在\[vmin, vmax\]范围内。

:param v: 输入值。

:param vmin: 最小值。

:param vmax: 最大值。

:return: 限制后的值。

\"\"\"

return vmin if v \< vmin else (vmax if v \> vmax else v)

def lowpass(prev, cur, dt, tau):

\"\"\"

一阶低通滤波器。

:param prev: 上一次的值。

:param cur: 当前的值。

:param dt: 时间差。

:param tau: 时间常数。

:return: 滤波后的值。

\"\"\"

a = dt/(tau+dt) if dt \> 0 else 1.0

return (1-a)\*prev + a\*cur

def deadband(e, db):

\"\"\"

死区函数，当误差小于死区时，返回0，否则返回误差减去死区值。

:param e: 误差。

:param db: 死区值。

:return: 处理后的误差。

\"\"\"

return 0.0 if abs(e) \<= db else (e - math.copysign(db, e))

\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-- 误差预处理
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

s\[\"lp_ex\"\] = lowpass(s\[\"lp_ex\"\], error_x, dt, s\[\"tau_err\"\])
\# 低通滤波处理X方向误差

s\[\"lp_ey\"\] = lowpass(s\[\"lp_ey\"\], error_y, dt, s\[\"tau_err\"\])
\# 低通滤波处理Y方向误差

ex = deadband(s\[\"lp_ex\"\], s\[\"db_x\"\]) \# 应用死区处理X方向误差

ey = deadband(s\[\"lp_ey\"\], s\[\"db_y\"\]) \# 应用死区处理Y方向误差

\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-- 到达统计
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

if abs(s\[\"lp_ex\"\]) \<= s\[\"hit_x\"\] and abs(s\[\"lp_ey\"\]) \<=
s\[\"hit_y\"\]:

\# 如果误差在到达阈值内，增加连续命中计数

s\[\"hit_cnt\"\] = min(s\[\"hit_cnt\"\] + 1, s\[\"hit_need\"\])

else:

\# 否则重置连续命中计数

s\[\"hit_cnt\"\] = 0

\# \-\-\-\-\-\-\-\-\-\-\-\-\-\-\-- 丢失保护
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

if (t - s\[\"last_det_ts\"\]) \> s\[\"lost_timeout\"\]:

\# 如果目标丢失超时，停止无人机

cmd = (0.0, 0.0, 0.0, 0.0)

else:

\# ========== 阶段 1：先 yaw 对准 ==========

if s\[\"phase\"\] == \"YAW_ALIGN\":

\# 计算偏航角速度

yawrate = clamp(s\[\"K_yaw\"\] \* ex, -s\[\"yaw_max\"\],
s\[\"yaw_max\"\])

vx = 0.0 \# 对准阶段不推进（也可以给很小的前进速度）

vy = 0.0

vz = 0.0

\# 判定是否对准：\|ex\| \< 阈值 且 连续满足

if abs(s\[\"lp_ex\"\]) \<= s\[\"yaw_align_tol\"\]:

s\[\"yaw_hold_cnt\"\] += 1

else:

s\[\"yaw_hold_cnt\"\] = 0

if s\[\"yaw_hold_cnt\"\] \>= s\[\"yaw_hold_need\"\]:

s\[\"phase\"\] = \"APPROACH\" \# 转入推进阶段

cmd = (vx, vy, vz, yawrate)

\# ========== 阶段 2：朝向目标推进 ==========

else: \# \"APPROACH\"

\# 若偏航又变大，退回对准阶段

if abs(s\[\"lp_ex\"\]) \> 1.5 \* s\[\"yaw_align_tol\"\]:

s\[\"phase\"\] = \"YAW_ALIGN\"

\# 立即给一次对准指令（可选）

yawrate = clamp(s\[\"K_yaw\"\] \* ex, -s\[\"yaw_max\"\],
s\[\"yaw_max\"\])

cmd = (0.0, 0.0, 0.0, yawrate)

else:

\# 偏航微调

yawrate = clamp(s\[\"K_yaw\"\] \* ex, -s\[\"yaw_max\"\],
s\[\"yaw_max\"\])

\# 将 ey 映射为俯仰方向角 alpha（X--Z 平面方向）

alpha = math.atan(ey / s\[\"ay\"\]) if s\[\"ay\"\] != 0 else 0.0

alpha = clamp(alpha, -s\[\"alpha_max\"\], s\[\"

#### **3.2.4 main.py代码解析**

用Visual Studio
Code软件打开[1.软件在环实验\\ServerFile\\main.py](1.软件在环实验/ServerFile/main.py)文件，这段代码是主函数，用于整合Smolagents框架和火山引擎LLM，实现无人机（UAV）的智能控制。它初始化BodyCommMavlink类以建立Mavlink通信，获取无人机列表和数量，然后创建OpenAI_APIs类的实例，传入功能函数以实现目标检测、接近目标等操作。最后，调用Main_Control方法启动交互模式，用户可通过自然语言输入指令，模型生成并执行控制代码。

main.py代码解析：

\# 主函数：利用Smolagents与火山引擎LLM实现UAV_Agent

import time

from OpenAI_api_Mavlink_Agent import OpenAI_APIs

from Communication_Mavlink import BodyCommMavlink

if \_\_name\_\_ == \'\_\_main\_\_\':

\# 创建BodyCommMavlink类的实例，初始化无人机的Mavlink通信

Comm_api = BodyCommMavlink()

\# 暂停5秒，确保通信初始化完成

time.sleep(5)

\# 获取无人机列表、无人机数量和坐标偏移量

MavList, VehilceNum, Error2UE4Map = Comm_api.GetBodyMavList()

\# 创建OpenAI_APIs类的实例，传入无人机列表、数量和功能函数

chat_api = OpenAI_APIs(

MavList,

VehilceNum,

Comm_api.detect_dino, \# 目标检测函数

Comm_api.approachObjective, \# 接近目标函数

Comm_api.look, \# 前置摄像头图像处理函数

Comm_api.search_object \# 搜索目标函数

)

\#
启动主控制逻辑，用户可以通过自然语言输入指令，模型生成相应的控制代码并执行

chat_api.Main_Control()

\'\'\'

\# 注释部分：用于测试detect_dino方法的性能

print(\"start thread_comm\")

while True:

start_time = time.time()

\# 调用detect_dino方法，检测目标为\"airplane\"

Comm_api.detect_dino(\"airplane\")

\# 打印每次调用的时间

print(time.time() - start_time)

\# time.sleep(5)

\'\'\'

### **实验说明**

（1）
单机控制时要非常注意相应的状态量是与飞机严格对应的，否则将导致程序紊乱。

（2）进行实验前，需要熟悉基本的控制理论，并认真阅读实验操作步骤。

4.  **实验的具体内容与步骤**

无人机视觉语言控制软件在环仿真实验主要包含如下步骤。

步骤一：配置实验环境。

步骤二：运行RflyUdpMavlinkRealSim.bat一键启动仿真。

步骤三：运行main.py控制飞机。

步骤四：结束程序。

详细内容参见[1.软件在环实验\\6-7软件在环仿真.pdf](1.软件在环实验/6-7软件在环仿真.pdf)的第4节"软件在环实验步骤"。

5.  **实验效果预览**

![图 1飞机识别并靠近蓝色小球指令](media/image8.emf)

6.  **思考与分析**

详细内容参见[1.软件在环实验\\6-7软件在环仿真.pdf](file:///C:\Users\admin\Desktop\Git_swarm\swarmcourse\swarmcourse\模块6-大模型控制\实验6-7_无人机视觉语言控制实验\1.软件在环实验\6-7软件在环仿真.pdf)的第5节"常见问题"。

# 实验6-7：无人机视觉语言控制软件在环仿真实验

## 1. 项目概述

### 1.1 实验背景

本实验利用软件在环(SIL)仿真技术研究无人机飞行控制策略。通过Python大语言模型自动代码生成对无人机进行控制，实现无人机通过自然语言指令识别不同颜色的小球并靠近的目标。

**核心技术栈:**
- 飞思集群仿真平台 (RflySim)
- MAVLink通信协议
- YOLO (目标检测)
- SmolAgents (代码化智能体)
- 火山引擎大语言模型 (deepseek-v3)

### 1.2 实验目标

1. 理解单机控制基本概念，熟悉单机接口配置
2. 掌握通过Python大语言模型自动生成控制指令，使无人机识别并靠近不同颜色的小球
3. 学会使用飞思集群仿真平台完成单机软件在环仿真

---

## 2. 系统架构

### 2.1 整体流程

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  自然语言指令    │ --> │  SmolAgents    │ --> │ 生成Python代码  │
│  (如"找红色球")  │     │   大模型决策    │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                                                          v
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  状态反馈(闭环)  │ <-- │  PX4MavCtrl    │ <-- │   执行代码      │
│                 │     │  发送控制指令   │     │ 调用感知/控制接口│
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ^                                                │
        │                                                v
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  无人机执行动作  │ <-- │ MAVLink通信     │ <-- │     YOLO       │
│ (旋转、逼近目标) │     │ 获取无人机状态  │     │   目标检测      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 2.2 三层架构

| 层级 | 功能 | 关键组件 |
|------|------|----------|
| **感知层** | 获取环境和无人机状态 | MAVLink通信、YOLO目标检测、LLM图像理解 |
| **决策层** | 将自然语言转换为控制策略 | SmolAgents框架、火山引擎LLM |
| **控制层** | 执行具体控制指令 | PX4MavCtrl、NED坐标系位置/速度控制 |

---

## 3. 核心模块详解

### 3.1 坐标系定义与转换

**NED坐标系** (无人机本地坐标系):
- N (North): 北向
- E (East): 东向
- D (Down): 向下

**UE4全局坐标系** (仿真环境):
- X: 东向
- Y: 北向
- Z: 向上

**坐标转换公式:**
```
Error2UE4Map = [
    -(GlobalPos_X - NED_X),  # X轴偏移
    -(GlobalPos_Y - NED_Y),  # Y轴偏移
    -(GlobalPos_Z - NED_Z)   # Z轴偏移 (注意方向相反)
]
```

### 3.2 YOLO 目标检测

**功能:** 实时目标检测，通过预训练模型识别图像中的目标

**处理流程:**
1. 图像预处理: 调整尺寸(640x640) -> 归一化 -> 格式转换
2. 模型推理: 输入图像，输出边界框、类别和置信度
3. 后处理: NMS非极大值抑制去除重叠框

**关键参数:**
```python
CONF_THRESHOLD = 0.25  # 置信度阈值，低于此值的检测框被过滤
NMS_THRESHOLD = 0.45   # NMS阈值，控制重叠框的合并程度
```

**YOLO到文本的映射:**
由于YOLO输出的是类别ID，需要通过映射表转换为文本:
```python
CLASS_NAMES = {0: "red ball", 1: "blue ball", 2: "yellow ball", 3: "airplane"}
```

**输出格式:**
- `obj_list`: 检测到的目标名称列表
- `obj_locs`: 边界框坐标 [x1, y1, x2, y2]
- `obj_logits`: 置信度分数
- `img_with_box`: 带标注框的图像

### 3.3 SmolAgents 代码化智能体

**核心功能:** 将自然语言指令转化为可执行的Python代码

**工作流程:**
1. 输入解析: 用户自然语言指令 -> 任务描述
2. 代码生成: LLM根据系统提示生成Python代码
3. 代码执行: 提取代码块并执行
4. 结果反馈: 将执行结果返回给LLM调整策略

**系统提示模板关键要素:**
- 可用函数说明: `detect_yolo()`, `approachObjective()`, `search_object()`, `look()`
- 控制逻辑约束: 偏航对齐后逼近、误差阈值判断等
- 安全保护: 目标丢失检测、超时处理

### 3.4 目标逼近控制算法 (approachObjective)

**两阶段控制策略:**

#### 阶段1: YAW_ALIGN (偏航对准)
- 仅调整偏航角，不进行位置移动
- 偏航角速度计算: `yawrate = K_yaw * error_x` (限幅±30°/s)
- 对准条件: |error_x| ≤ yaw_align_tol (25像素)，连续满足3次

#### 阶段2: APPROACH (朝向目标推进)
- 偏航微调: 持续根据error_x调整偏航
- 速度分解:
  ```
  alpha = atan(ey / ay)  # 俯仰方向角
  vx = v_nom * cos(alpha)  # 前进速度
  vz = v_nom * sin(alpha)  # 垂直速度 (FRD坐标系)
  ```
- 速度限幅: v_min=0.05, v_max=1.0, vz_max=0.35

**误差处理:**
- 低通滤波: `lp_error = (1-a)*prev + a*cur`，时间常数tau=0.5s(CPU)或0.25s(GPU)
- 死区处理: |error| ≤ 5像素时视为0，避免抖动
- 到达判断: |error| ≤ 1像素，连续满足3次认为到达目标

**安全保护:**
- 目标丢失超时: lost_timeout = max(3.0/det_fps, 1.5)秒
- 阶段回退: 若偏航误差超过阈值，从APPROACH退回YAW_ALIGN

---

## 4. 代码文件结构

```
ServerFile/
├── main.py                          # 主程序入口
├── Communication_Mavlink.py         # MAVLink通信与目标检测核心类
├── OpenAI_api_Mavlink_Agent.py      # SmolAgents智能体封装
├── volcEngineLLM.py                 # 火山引擎LLM API封装
├── Description.py                   # 提示词模板定义
├── Coordinate_Transformation.py     # 坐标转换工具
├── VisionCaptureApi.py              # 视觉捕获API
├── PX4MavCtrlV4.py                  # PX4无人机控制接口
├── ReqCopterSim.py                  # 仿真环境通信
│
├── weights/                       # 自定义目标检测模型权重
│   ├── best.pt                   # 自定义目标检测模型权重
│   └── ...
│
└── .asset/                          # 临时资源文件
```

### 4.1 核心类说明

#### BodyCommMavlink (Communication_Mavlink.py)

**职责:** 无人机通信、目标检测、图像理解、目标逼近控制

**关键方法:**

| 方法名 | 功能 | 参数 | 返回值 |
|--------|------|------|--------|
| `__init__` | 初始化MAVLink连接、加载YOLO模型、启动图像捕获 | - | - |
| `detect_yolo` | 使用YOLO检测目标 | object_names: str | obj_list, obj_locs, obj_logits, img |
| `search_object` | 旋转搜索目标(每次40°) | object_names: str | bool (是否找到) |
| `look` | 使用LLM理解当前图像内容 | - | content: str |
| `approachObjective` | 控制无人机逼近目标 | error_x, error_y | - |

**成员变量:**
```python
self.MavList        # 无人机控制器列表
self.VehilceNum     # 无人机数量
self.Error2UE4Map   # 坐标系偏移量
self.yolo_model    # YOLO模型实例
self.vis            # VisionCaptureApi实例(图像捕获)
self.llm_client     # 火山引擎LLM客户端
```

#### OpenAI_APIs (OpenAI_api_Mavlink_Agent.py)

**职责:** 智能体交互、代码生成与执行

**关键方法:**

| 方法名 | 功能 |
|--------|------|
| `__init__` | 初始化无人机列表、API密钥、功能函数引用 |
| `Agents_UAV` | 主交互循环: 接收指令 -> 生成代码 -> 执行代码 |
| `execute_generated_code` | 清理并执行LLM生成的Python代码 |
| `GetHistrory` | 记录对话历史 |

**功能函数注入:**
```python
self.detect_function         # Comm_api.detect_yolo
self.approachObjective_function  # Comm_api.approachObjective
self.look_function           # Comm_api.look
self.search_object_function  # Comm_api.search_object
```

#### VolcEngineFakeHFModel (volcEngineLLM.py)

**职责:** 将火山引擎API包装为Hugging Face格式，供SmolAgents使用

**配置:**
```python
api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
model_id = "deepseek-v3-250324"
```

---

## 5. 实验执行流程

### 5.1 环境配置要求

**软件依赖:**
- Python 3.8+
- PyTorch (CPU/GPU)
- OpenCV
- NumPy
- SmolAgents
- OpenAI Python SDK

**模型文件:**
- `weights/best.pt` - 自定义目标检测模型权重

**仿真环境:**
- 飞思集群仿真平台 (RflySim)
- UE4仿真场景

### 5.2 启动步骤

**步骤1: 启动仿真环境**
```bash
# 运行一键启动脚本
RflyUdpMavlinkRealSim.bat
```

**步骤2: 运行主程序**
```bash
cd ServerFile
python main.py
```

**步骤3: 输入自然语言指令**
```
请输入你的控制模式指令: 找到红色气球并靠近
```

**步骤4: 结束程序**
```
请输入你的控制模式指令: exit
# 或按 Ctrl+C
```

### 5.3 典型指令示例

```
"找到红色气球并靠近"
"搜索蓝色小球"
"查看当前视野中的目标"
"在视野中搜索飞机"
```

---

## 6. 关键算法细节

### 6.1 目标搜索算法 (search_object)

```
当前偏航角 -> 每次增加40° -> 设置新偏航角 -> 等待2秒 -> 检测目标
    ^                                               |
    |                                               |
    <---------------- 未找到则继续旋转 --------------
```

### 6.2 阶段式逼近控制

```
初始状态: YAW_ALIGN
    |
    v
[YAW_ALIGN阶段] --> 偏航误差|ex|<=25? --> 连续3次? --> 进入APPROACH
    |                                          |
    误差>25                                   否
    |                                          |
    v                                          v
调整偏航角                                保持YAW_ALIGN
    |                                          |
    <------------------------------------------

[APPROACH阶段] --> 偏航误差|ex|>37.5? --> 退回YAW_ALIGN
    |
    否
    v
同时调整偏航和推进速度
    |
    v
误差|ex|,|ey|<=1? --> 连续3次? --> 目标到达
```

### 6.3 速度合成公式

```python
# 将图像平面误差映射为三维空间速度
alpha = math.atan(ey / ay)          # 俯仰方向角 (ay=327, 控制敏感度)
alpha = clamp(alpha, -85°, +85°)     # 限制最大俯仰角

vx = v_nom * math.cos(alpha)        # 北向速度 (前进)
vz = v_nom * math.sin(alpha)        # 垂直速度 (下降/上升，FRD坐标系)

# 应用限幅
vx = clamp(vx, 0.05, 1.0)           # 速度范围
vz = clamp(vz, -0.35, +0.35)        # 垂直速度限幅
```

---

## 7. 常见问题与调试

### 7.1 状态量对应

**注意:** 单机控制时必须确保状态量与飞机严格对应，否则将导致程序紊乱。

**状态量说明:**
```python
mav.uavPosNED      # NED坐标系位置 [N, E, D]
mav.uavAngEular    # 欧拉角 [roll, pitch, yaw] (弧度)
mav.uavVelNED      # NED坐标系速度 [VN, VE, VD]
mav.uavGlobalPos   # UE4全局坐标 [X, Y, Z]
```

### 7.2 调试技巧

1. **检测帧率:** CPU环境下det_fps=0.4，GPU环境下det_fps=10
2. **图像保存:** `annotated_image.jpg` 保存带检测框的图像
3. **API调用日志:** 查看"火山方舟 API 调用成功/失败"输出
4. **AI计算时间:** 每次指令会输出AI生成代码耗时

### 7.3 性能优化

- GPU环境下启用CUDA加速目标检测
- 调整`CONF_THRESHOLD`和`NMS_THRESHOLD`平衡检测精度与速度
- 修改`ay`参数调整俯仰敏感度

---

## 8. 扩展与改进方向

1. **控制算法优化:** 引入PID控制替代简单的比例控制
2. **提示工程:** 优化SmolAgents系统提示，提高代码生成质量
3. **多机协同:** 扩展至多机编队控制
4. **语义理解:** 增强LLM对复杂指令的理解能力
5. **安全机制:** 增加碰撞检测、边界保护等安全功能

---

## 9. 参考资料

- [RflySim 文档](https://rflysim.com/)
- [YOLOv8 文档](https://docs.ultralytics.com/)
- [SmolAgents 文档](https://huggingface.co/docs/smolagents)
- [MAVLink 协议](https://mavlink.io/)
- [PX4 飞行控制](https://px4.io/)

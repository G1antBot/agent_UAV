# agent_UAV

本实验通过Mavlink通信协议、开放词汇目标检测（Grounding DINO）、代码化智能体（SmolAgents）与火山引擎大语言模型（LLM）的深度融合，实现了"自然语言指令→大模型决策→无人机控制→环境感知"的闭环仿真。实验核心是将人类的自然语言需求转化为无人机可执行的控制逻辑，本质是跨模态感知（视觉-文本）+代码化决策+模型预测控制的综合应用。以下从系统架构、关键模块原理、闭环逻辑三个层面展开说明，并结合公式与数学推导深化理解。

---

## 系统架构 (System Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│              自然语言任务输入  (Natural Language Task)            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────▼────────────────┐
            │  火山方舟 LLM (VolcanoLLMClient) │
            │  任务理解 & 子目标规划            │
            └───────────────┬────────────────┘
                            │  结构化计划
            ┌───────────────▼────────────────┐
            │  SmolAgents CodeAgent           │
            │  (UAVAgent)                     │
            │  计划 → Python 工具调用          │
            └──────────┬──────────┬──────────┘
                       │ 控制指令  │ 感知请求
                       ▼          ▼
          ┌─────────────────┐  ┌──────────────────────┐
          │ MAVLink控制器    │  │  Grounding DINO       │
          │ (或 Simulator)  │  │  (开放词汇目标检测)     │
          └────────┬────────┘  └──────────┬────────────┘
                   │ 遥测数据               │ 检测结果
                   └──────────┬────────────┘
                              │ 反馈信息
                              ▼
            ┌───────────────────────────────┐
            │  VolcanoLLMClient (再评估)     │
            │  生成下一个子目标               │
            └───────────────────────────────┘
```

### 模块说明

| 模块 | 文件路径 | 功能 |
|------|----------|------|
| **MAVLink控制器** | `src/uav/mavlink_controller.py` | 封装 pymavlink，提供 arm/takeoff/move_to 等高级原语 |
| **UAV 仿真器** | `src/uav/simulator.py` | 纯内存仿真，无需真实硬件，用于测试与离线演示 |
| **Grounding DINO** | `src/perception/grounding_dino.py` | 开放词汇目标检测，输入图片+文本提示→输出边框+置信度 |
| **火山方舟 LLM** | `src/llm/volcano_engine.py` | 调用火山引擎 Ark API（兼容 OpenAI 格式），维护对话历史 |
| **UAV Agent** | `src/agent/uav_agent.py` | SmolAgents CodeAgent，注册10个UAV工具，驱动闭环决策 |
| **Orchestrator** | `src/orchestrator.py` | 串联所有子系统，管理连接、任务执行与断开 |
| **主入口** | `src/main.py` | CLI 入口，支持命令行参数与 `.env` 配置 |

---

## 关键模块原理 (Key Module Principles)

### 1. MAVLink 通信协议

MAVLink（Micro Air Vehicle Link）是一种轻量级的消息编组协议，专为资源受限的嵌入式系统设计。

**消息结构：**

```
[STX | LEN | SEQ | SYS | COMP | MSG_ID | PAYLOAD | CRC]
```

本项目中使用的关键 MAVLink 命令：

| 指令 | MAV_CMD | 用途 |
|------|---------|------|
| 解锁 | `MAV_CMD_COMPONENT_ARM_DISARM` | 激活电机 |
| 起飞 | `MAV_CMD_NAV_TAKEOFF` | 上升至目标高度 |
| 降落 | `MAV_CMD_NAV_LAND` | 就地降落 |
| 导航 | `SET_POSITION_TARGET_GLOBAL_INT` | 飞往GPS坐标 |
| 速度 | `MAV_CMD_DO_CHANGE_SPEED` | 设置空速 |

**坐标系：** `MAV_FRAME_GLOBAL_RELATIVE_ALT_INT`
- 纬度/经度：1e-7 度整数
- 高度：相对起飞点的相对高度（米）

### 2. Grounding DINO 目标检测

Grounding DINO（Liu et al., 2023）将**视觉 Transformer（Swin）**与**语言模型（BERT）**通过特征增强模块融合，实现开放词汇检测。

**双模态匹配公式：**

给定图像特征 `F_v ∈ R^(N_v × d)` 和文本特征 `F_t ∈ R^(N_t × d)`，交叉注意力融合为：

```
Attn(Q, K, V) = softmax(Q·Kᵀ / √d) · V
```

其中 Q 来自图像 token，K, V 来自文本 token，形成**视觉-语言对齐**。最终使用匈牙利算法（Hungarian matching）完成二分匹配。

### 3. SmolAgents 代码化决策

SmolAgents 采用"Think → Code → Execute"循环：

```
输入任务
  → LLM 生成 Python 代码（工具调用）
  → 安全沙箱执行代码
  → 观察结果
  → 迭代（最多 max_steps 轮）
  → 输出最终结果
```

本项目注册了10个UAV专用工具：`arm`, `disarm`, `takeoff`, `land`,
`return_to_launch`, `move_to`, `set_airspeed`, `hover`,
`detect_objects`, `get_telemetry`。

### 4. 火山引擎大语言模型

接入**火山方舟（Ark）**推理平台，使用 Doubao 系列模型。API 与 OpenAI Chat Completions 格式兼容：

```python
from volcenginesdkarkruntime import Ark
client = Ark(api_key="...", base_url="https://ark.cn-beijing.volces.com/api/v3")
response = client.chat.completions.create(
    model="ep-xxxx-yyyy",   # 推理接入点 ID
    messages=[...],
)
```

---

## 闭环逻辑 (Closed-Loop Logic)

```
time=0: 用户输入 → "搜索目标并返航"
   ↓
time=1: LLM 规划 → [起飞, 移动到搜索区, 目标检测, 返航]
   ↓
time=2: Agent 执行 arm() → takeoff(10) → move_to(...)
   ↓
time=3: detect_objects("person . vehicle") → Grounding DINO
   ↓
time=4: 检测结果反馈给 LLM → "发现2人，坐标(x,y)"
   ↓
time=5: LLM 决策 → return_to_launch()
   ↓
time=6: 任务完成，生成报告
```

---

## 快速开始 (Quick Start)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

创建 `.env` 文件：

```env
VOLC_ACCESSKEY=your_volcano_engine_api_key
VOLC_ENDPOINT_ID=ep-xxxxxxxx-xxxxx
```

### 运行仿真

```bash
# 使用内置仿真器（无需真实硬件）
python -m src.main "起飞到10米，向北飞行100米，检测是否有人，返航"

# 指定配置文件
python -m src.main --config configs/config.yaml "执行侦察任务"

# 连接真实飞控（需要 MAVLink 连接）
python -m src.main --no-simulator "起飞到5米悬停"
```

### 运行测试

```bash
pytest tests/ -v
```

---

## 项目结构 (Project Structure)

```
agent_UAV/
├── configs/
│   └── config.yaml          # 系统配置（LLM、MAVLink、检测参数）
├── src/
│   ├── uav/
│   │   ├── mavlink_controller.py   # MAVLink 高级封装
│   │   └── simulator.py            # 内存UAV仿真器
│   ├── perception/
│   │   └── grounding_dino.py       # Grounding DINO 检测器
│   ├── llm/
│   │   └── volcano_engine.py       # 火山方舟 LLM 客户端
│   ├── agent/
│   │   └── uav_agent.py            # SmolAgents 代码智能体
│   ├── orchestrator.py             # 闭环协调器
│   └── main.py                     # CLI 主入口
├── tests/
│   ├── test_simulator.py
│   ├── test_grounding_dino.py
│   ├── test_volcano_engine.py
│   └── test_orchestrator.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## License

MIT © 2026 G1antBot

# 基于大模型的无人机识别打击控制算法

## 1. 项目概述

### 1.1 当前版本定位

本项目当前主链路是“自然语言任务 -> 子句级模板拦截/大模型生成 -> 视觉伺服控制执行 -> 日志回传”的闭环，实现了在 RflySim 软件在环环境中的目标搜索、朝向、靠近与打击动作。

当前实现重点是：
1. 分层控制：感知层、决策层、控制层解耦。
2. 子句拦截：对高频语义优先走模板执行，减少纯生成式控制的不确定性。
3. 朝向收敛：先对准再推进，保障靠近与打击过程稳定。

说明：ROS 不是当前代码主链路依赖，当前主流程基于 RflySim SDK + MAVLink + Python 运行时。

### 1.2 核心技术栈

- RflySim（SIL 仿真）
- MAVLink / PX4 控制接口
- YOLOE（`weights/best.pt`）目标检测
- SmolAgents + 火山引擎 LLM（`deepseek-v3-250324`）
- `runtime_logger` 运行日志体系

---

## 2. 系统架构

### 2.1 主链路流程

```
自然语言指令
    -> 子句拆分与拦截
        -> 模板执行(位移/转向/搜索/靠近/朝向)
        -> 或 AI 生成代码执行(复杂语义)
            -> 感知反馈(detect_yolo / look)
                -> 控制下发(SendPosNED / SendVelFRD)
                    -> 任务结果与日志回传
```

### 2.2 三层职责

| 层级 | 职责 | 当前实现 |
|------|------|----------|
| 感知层 | 目标检测与视觉状态读取 | `detect_yolo`、`look`、检测结果缓存与保存 |
| 决策层 | 指令切分、模板路由、生成式兜底 | `_split_task_clauses` + `_parse_*_clause` + `_run_agent_for_clause` |
| 控制层 | 动作执行与收敛控制 | `face_objective_to_target`、`approach_objective_to_target`、`strike_objective_to_target` |

---

## 3. 当前核心机制

### 3.1 子句拦截与执行优先级

在 `Agents_UAV` 主循环中，输入会先按“然后/并且/标点”拆分为子句，再按优先级尝试模板匹配：

1. 位移模板（机体系前后左右上下）
2. 转向模板（左转/右转）
3. 搜索模板（quick/all）
4. 靠近模板
5. 朝向模板（只转向不前进）
6. 未命中模板时，回退到 AI 代码生成执行

这样可将“稳定可规则化”的动作固定在模板层，把复杂语义留给大模型，降低控制漂移和执行歧义。

### 3.2 搜索策略（quick / all）

`search_object_detail` 支持两种模式：

- `quick`：先检测当前视野，未命中则按 40° 步进旋转，命中即停。
- `all`：完整旋转一圈后统计目标数量，并输出相对朝向角聚类结果。

同时支持目标名归一化（中英文别名映射），减少“同义词导致漏检”的问题。

### 3.3 朝向收敛（原地对准）

`face_objective_to_target` 的目标是“只转向，不推进”：

1. 先搜索目标。
2. 循环检测目标框中心误差。
3. 调用 `faceObjectiveOnly` 输出偏航角速度。
4. 连续满足对准阈值后判定收敛。

该机制用于在靠近和打击前稳定机头方向，降低后续动作的横向误差。

### 3.4 分层逼近（YAW_ALIGN -> APPROACH）

`approachObjective` 采用两阶段伺服：

1. `YAW_ALIGN`：仅调偏航，直到水平误差连续满足阈值。
2. `APPROACH`：保持偏航微调，同时根据图像误差分解前进/垂直速度。

并包含低通滤波、死区、阶段回退、目标丢失超时和指令节流，核心目的是“先对准，再前进”，保证靠近过程可收敛。

`approach_objective_to_target` 在上层封装了“搜索 -> 循环检测 -> 伺服靠近 -> 停稳判定”的完整任务闭环。

### 3.5 打击动作（高级模板能力）

`strike_objective_to_target` 在搜索与对准后执行前冲穿越并停稳，属于高风险动作能力，默认作为高级动作由任务语义触发。

---

## 4. 代码结构（与当前仓库一致）

```
ServerFile/
├── main.py                        # 程序入口，组装通信与智能体
├── Communication_Mavlink.py       # 感知与控制核心实现
├── OpenAI_api_Mavlink_Agent.py    # 子句拦截、模板执行、AI生成兜底
├── Description.py                 # 系统提示词与能力约束
├── Coordinate_Transformation.py   # 机体系->NED 坐标转换
├── volcEngineLLM.py               # 火山引擎模型适配
├── runtime_logger.py              # 日志封装
├── Config.json                    # 运行配置
├── weights/
│   └── best.pt                    # 当前检测主权重
├── logs/
└── saved_detections/
```

说明：视觉采集与飞控底层接口通过 RflySim SDK 路径注入（`sys.path.append(...)`），不在本仓库内维护。

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
2. **图像保存:** 优先查看 `saved_detections/` 目录下的检测图，也可以用缓存图回看最近一次结果
3. **API调用日志:** 查看运行日志里模型请求、检测结果和执行异常信息
4. **AI计算时间:** 每次指令会输出 AI 生成代码耗时，方便判断是否存在卡顿

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

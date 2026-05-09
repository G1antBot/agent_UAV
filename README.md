# 基于大模型的无人机视觉语言控制算法与评测系统

## 1. 项目概述

### 1.1 当前版本定位

本项目实现了一个基于大语言模型（LLM）的无人机视觉语言控制闭环系统：“自然语言任务 -> 子句级拆分与意图拦截 -> 规则模板 / LLM代码生成 -> 视觉伺服执行 -> 状态监控与反馈”。
系统支持 RflySim 软件在环仿真（SIL）、室内动捕实飞（Mocap）和室外光流/GPS实飞（Optical）三种模式。

当前版本的核心特色在于：
1. **混合决策架构**：采用“硬规则层优先，生成式LLM兜底”的混合策略，将确定性动作（基础位移、搜索、靠近、返回起飞点）固化，复杂时序与分支动作交由LLM代码沙箱执行。
2. **多线程安全看门狗**：引入双线程架构（看门狗输入监听 + Agent Worker 顺序执行），保证在 LLM 或底层循环阻塞时，急停指令能够被毫秒级响应。
3. **全局防碰撞与飞行包线保护**：底层的动作限幅、空间地理围栏（水平半径）、贴地与超高防碰撞保护。
4. **完整的学术评测体系**：包含自动化评测脚本（Exp1: 安全性与急停消融，Exp2: 语义保真度评估）与 Nature/IEEE 风格的数据可视化出图代码，满足毕业设计与论文撰写要求。

### 1.2 核心技术栈

- 飞控接口：MAVLink / PX4 (基于 RflySim SDK)
- 视觉检测：YOLOE（`weights/best.pt`）
- 大语言模型：SmolAgents + 火山引擎 API（`deepseek-v3-250324`）
- 室内定位：OptiTrack 动捕 UDP 组播接收（Motive）
- 评测与可视化：Pandas, Matplotlib
- 日志系统：`runtime_logger` 运行与追踪日志体系

---

## 2. 系统架构

### 2.1 主链路流程

```
自然语言指令
  ├── [Watchdog 线程] 拦截“急停”等最高优先级指令 -> 立即清空队列、速度归零
  └── [Worker 线程] 
        -> 复杂指令判定 (自适应/条件分支) -> 交由 LLM
        -> 或 子句拆分
            -> 模板拦截 (位移/转向/搜索/靠近/朝向/返航/条件分支)
            -> 或 LLM 生成代码沙箱执行
                -> 视觉感知 (detect_yolo / look)
                -> 飞控下发 (含安全包线拦截与自动限幅)
                    -> 任务结果提取与日志持久化
```

### 2.2 三层职责与安全机制

| 层级 | 职责 | 关键组件与机制 |
|------|------|----------|
| **感知层** | 目标检测与全局状态读取 | `detect_yolo`、`look`、`MocapClient` (实飞位姿桥接) |
| **决策层** | 指令切分、模板路由、LLM沙箱 | `OpenAI_api_Mavlink_Agent.py`：双线程管理、子句解析、代码生成、条件模板 |
| **控制层** | 动作伺服收敛与闭环执行 | `Communication_Mavlink.py`：阶段式逼近(`APPROACH`)、原地朝向、打击穿越、固定位移 |
| **安全层** | 防止撞击、乱飞与指令超时 | 全局速度与限位卡控（`guarded_send_vel_frd`）、高度墙(`enable_alt_guard`)、空间围栏(`enable_space_fence`) |

---

## 3. 当前核心机制

### 3.1 混合决策与子句拦截优先级

对于普通的组合动作（如“向前飞2米然后找红色气球并靠近”），系统会按连接词切分子句，并按如下优先级尝试模板匹配：
1. **最高级安全**：急停、退出指令（看门狗线程直接拦截）。
2. **条件分支模板**：包含“如果有...有的话...没有就...”的简单逻辑路由。
3. **位移模板**：机体系前后左右上下（解析距离数值）。
4. **转向模板**：原地左/右转（解析角度）。
5. **搜索模板**：支持 `quick`（命中即停）和 `all`（旋转一圈统计）。
6. **靠近模板**：基于视觉伺服的追踪靠近。
7. **朝向模板**：原地对准目标不前进。
8. **返航模板**：飞回起飞点。
9. **LLM 兜底**：未命中或被判定为复杂不可拆分语义时，统一交由大模型生成调用API的执行代码。

### 3.2 阶段式逼近与目标打击

- **靠近 (`approach_objective_to_target`)**：
  分阶段状态机设计。阶段1 `YAW_ALIGN`：原地微调偏航，直至连续3次偏航误差低于阈值；阶段2 `APPROACH`：同时维持偏航并根据目标中心像素误差调度纵向与垂向速度。包含低通滤波、死区以及丢失防抖保护。当目标边框占据视野达到设定比例且居中时判定为靠近停稳。
- **打击 (`strike_objective_to_target`)**：
  高级战斗/穿越动作。先远距离引导对准并高速靠近（根据目标大小动态排程速度），当逼近极限距离时进入“终端穿越”盲飞冲刺段，随后急刹悬停，模拟击中或穿越。

### 3.3 运行模式体系 (Config.json)

配置 `run_mode` 决定底层数据源：
- `sim`：软件在环仿真，直接调用 RflySim 的 UE4 内存图像接口获取视觉数据。
- `real_mocap`：基于室内 OptiTrack 的实飞模式，建立 `MocapClient` UDP接收线程桥接刚体坐标，通过 `SendVisionPosition` 注入飞控；图像从真实相机 (`opencv`) 实时读取。
- `real_optical`：室外或光流定点模式，不依赖动捕，图像同为真实相机。

仿真和实飞均支持一键开启 **YOLO实时检测预览窗**（`auto_start`）。

---

## 4. 代码结构说明

```
├── ServerFile/
│   ├── main.py                        # 主控制节点，组装通信、检测、桥接与代理
│   ├── OpenAI_api_Mavlink_Agent.py    # 智能体核心：双线程输入、指令切分、LLM沙箱、模板匹配
│   ├── Communication_Mavlink.py       # 底层飞控、安全拦截器、视觉伺服实现
│   ├── Description.py                 # 大语言模型系统提示词与能力约束定义
│   ├── Config.json                    # 核心配置文件 (运行模式、相机、安全开关等)
│   ├── MocapClient.py                 # OptiTrack 动捕 UDP 桥接客户端
│   ├── simple_output_window.py        # Tkinter 极简指令与结果实时反馈窗口
│   ├── volcEngineLLM.py               # 火山引擎 DeepSeek API 适配与格式过滤
│   ├── Coordinate_Transformation.py   # 机体->NED 欧拉角坐标转换工具
│   ├── runtime_logger.py              # 统一日志管理器 (持久化到 logs/)
│   ├── test_exp1.py                   # [实验一] 架构消融与急停安全性评测
│   ├── test_exp2.py                   # [实验二] 复杂语义指令解析与成功率评测
│   ├── visualize_results.py           # 学术论文数据可视化绘图脚本
│   ├── weights/best.pt                # YOLO 目标检测权重
│   ├── logs/                          # 运行日志与代码沙箱回溯目录
│   └── saved_detections/              # 相机拍摄与检测结果图缓存目录
```

---

## 5. 实验启动与运行流程

### 5.1 环境配置要求
- Python 3.8+ (推荐 Python 3.12)
- 飞思 RflySim 集群仿真平台 (需保证网络联通与 UE4 渲染)
- `pip install -r requirements.txt` (包含 `smolagents`, `ultralytics`, `openai`, `optirx`, `pandas`, `matplotlib`)

### 5.2 启动主系统 (交互控制)

1. 配置 `ServerFile/Config.json`（设定 `run_mode` 为 `sim` 或 `real_mocap`）。
2. 若为仿真，先运行根目录的一键启动脚本 `RflyUdpMavlinkRealSim.bat` 开启 UE4 和 PX4SITL。
3. 启动 Python 主程序：
```bash
cd ServerFile
python main.py
```
4. 在终端出现“系统已就绪（双线程模式）”后，输入自然语言指令。例如：
   - `起飞到半米，然后找红色气球`
   - `靠近它，速度慢一点`
   - `看看四周有什么，有蓝球就靠近，否则返航`
   - `急停`

### 5.3 自动化实验评测 (为论文提供数据支撑)

**实验一：急停控制与飞行安全包线评测**
```bash
cd ServerFile
python test_exp1.py --trials 30 --stress-speed 0.8
```
用于对比硬规则架构与纯大模型架构在面临极限飞行动作时，发送“急停”指令的响应延迟、漂移距离和避撞成功率。

**实验二：多模态长序列指令理解与执行评测**
```bash
cd ServerFile
python test_exp2.py
```
读取 `md/experiment2_instruction_library.md` 中的学术指令集，从基础运动、多时序逻辑到条件分支等维度，记录解析成功率(PSR)、执行完成率(TCR)与语义一致性评分(SFS)。

**绘制实验结果学术图表**
```bash
python visualize_results.py
```
将自动读取 `logs/test_exp2/` 下的数据，生成符合 Nature / IEEE 标准的高清 `PNG` 和矢量 `PDF` 统计图（雷达图、热力图、盒须图等）。

---

## 6. 常见排障 (Troubleshooting)

1. **预检失败 / 提示“动捕位姿回调返回无效首包”**
   - 检查局域网内 Motive 的 Multicast 广播是否开启（默认 IP 239.255.42.99，端口 1511）。
   - 检查 `Config.json` 中的 `rigid_body_id` 是否与 Motive 中的飞机 ID 匹配。
2. **生成的代码报执行异常**
   - 检查 `logs/code/` 目录中最近的 Python 代码切片，查看模型是否使用了未注入的库或死循环。系统带有180秒看门狗超时机制。
3. **视觉伺服卡在 YAW_ALIGN 偏航阶段**
   - 可能是 GPU/CPU 推理延迟差异引起。若使用 CPU（FPS≈0.4），无人机会有明显的“走走停停”。系统现已加入 `yaw_align_timeout`，超时后会强制切到推进阶段。
4. **触发“空间围栏”或“高度保护”被迫返航**
   - 检查 `Config.json` 中的安全层限制（`max_radius_m`，`alt_ceiling_ned` 等），确保给定的测试距离没超过设定的物理限制墙。

---

## 7. 引用与参考

- [RflySim](https://rflysim.com/) - 无人机数字孪生仿真
- [OptiTrack Motive API](https://optitrack.com/) - 室内定位
- [SmolAgents](https://huggingface.co/docs/smolagents) - 轻量级代理框架
- 火山引擎 DeepSeek API - 后端大语言模型支持

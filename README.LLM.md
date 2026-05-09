# README.LLM

面向人类开发者与大语言模型代码助手的大纲化、高密度说明文档。
目标是让读者在最短时间内理解本仓库的**全链路代码结构、双线程并发模型、硬规则路由引擎、安全拦截器**及排障入口。

## 1. 核心模块与阅读顺序

建议按下面顺序阅读代码以快速建立心理模型：

1. **`ServerFile/Config.json`**：系统运行中枢配置（模式、动捕、相机、UI预览开关）。
2. **`ServerFile/main.py`**：启动总线。负责预检、挂载相机/动捕、挂载预览窗、初始化Agent架构并进入循环。
3. **`ServerFile/OpenAI_api_Mavlink_Agent.py`**：核心大脑。实现双线程架构（看门狗输入+Worker执行）、硬规则切分与匹配、LLM沙箱代码执行。
4. **`ServerFile/Communication_Mavlink.py`**：身体与小脑。封装Mavlink/PX4控制，实现带有死区和低通滤波的视觉伺服，以及**极其严格的底层运动安全包线拦截器**。
5. **`ServerFile/test_exp*.py` & `visualize_results.py`**：验证与学术输出代码。

## 2. 三种运行模式 (`run_mode`)

配置来源：`Config.json -> run_mode`

| 模式 | 图像源 | 飞控端IP | 外部位姿 (Mocap) |
|---|---|---|---|
| `sim` | UE4内存接口 (`VisionCaptureApi`) | RflySim自动获取 | 无 |
| `real_mocap` | 真实USB/UVC相机 (`opencv`) | `real_mavlink.target_ip` | 开启 `MocapClient` UDP接收，并通过 `SendVisionPosition` 注入 |
| `real_optical`| 真实USB/UVC相机 (`opencv`) | `real_mavlink.target_ip` | 无 (依赖光流/GPS) |

**注**：`sim`和`real_mocap`均支持一键启动带有检测框的UI实时预览（受`sim_preview`和`real_camera.preview`控制）。

## 3. 并发模型与指令流

### 3.1 双线程守护架构 (Agent Worker + Watchdog)
入口：`Agent.Main_Control()` -> `Agent.Agents_UAV()`

- **Watchdog-Input 线程**：
  - 阻塞式监听用户终端输入。
  - **急停判定**：正则匹配到“急停”等词汇，**不入队列**，直接调用底层的 `SendVelFRD(0,0,0,0)` 刹车，将全局 `is_interrupted` 置 `True`，并清空历史任务队列。
- **Agent-Worker 线程**：
  - 循环消费任务队列。
  - 如果被判定为**复杂语义**（如带修饰的连续动作、自适应条件），不拆分，直接交由 `deepseek-v3` 解析执行。
  - 如果是常规语句，按连接词拆分子句，进入硬规则路由。

### 3.2 路由与规则优先级
`_handle_hard_rules(clause)` 将拦截标准指令并直接下发Mavlink，跳过LLM推理以确保稳定和低延迟：
1. **条件分支** (`_handle_conditional_task`)："如果有气球就靠近，否则返航"
2. **基础位移**：解析“前后左右上下”及带小数点的米数。
3. **转向**：原地改变偏航角。
4. **搜索**：`quick` 或 `all`。
5. **靠近**：进入视觉伺服逼近循环。
6. **对准**：原地调整偏航对准。
7. **回起点**：读取起飞记录坐标并飞回。

只有全部无法匹配，才调用大模型（`_run_agent_for_clause` -> 建立安全沙箱执行Python代码）。

## 4. 安全保护系统 (Safety Guards)

**位置：** `Communication_Mavlink.py -> _install_motion_safety_guards()`

为了防止实验出现炸机事故以及限制大模型的幻觉代码，底层 `MavList[0]` 的所有移动和下发接口（如 `SendVelFRD`, `SendPosNED`）都被**拦截包装**。
- **动态限幅**：根据当前模式（搜索、靠近、打击），对速度和偏航角速度施加严格上限。
- **高度墙 (Alt Guard)**：下发高度若低于 `alt_floor_ned` (贴地) 会被强制拉起到安全高度；高于 `alt_ceiling_ned` 会被拦截并悬停。
- **空间围栏 (Space Fence)**：基于起飞点 (`_home_pos_ned`)，一旦下发的预测位置超出 `max_radius_m`，直接驳回该指令。
- **超时保护**：每个子句最多执行60/180秒，超时后飞控层指令自动作废，悬停。

## 5. 视觉伺服实现细节

- **`approach_objective_to_target`** (追踪靠近)：
  两级状态机 (`YAW_ALIGN` -> `APPROACH`)。通过低通滤波和死区解决画面的闪烁；当目标占据画面宽度的一定比例（如1/5），且目标中心落入像素阈值内时，发送全零速度停止。
- **`strike_objective_to_target`** (前冲打击)：
  由于近距离会触发丢目标/穿模死锁，采取“先引导逼近，最后阶段根据距离换算时间进行终端盲飞冲刺”的策略。

## 6. 学术实验与评测框架

本仓库不仅是工程控制代码，还内建了一套专门为写论文（如毕业设计或学术期刊）服务的评测与可视化工作流：

1. **`test_exp1.py`** (架构安全性验证)：
   自动控制无人机乱飞并施加扰动，然后在不同架构（硬规则 vs 纯LLM）下注入“急停”，评测并输出系统的最大漂移距离、响应延迟、稳定时间（`logs/test_exp1/`）。
2. **`test_exp2.py`** (智能体语义解析测试)：
   加载 `md/experiment2_instruction_library.md` 中数百条测试用例，在本地调用LLM并沙箱执行，记录任务完成率(TCR)、解析成功率(PSR)和非常细致的语义保真度(SFS，借鉴 ALFRED GC 指标)。
3. **`visualize_results.py`**：
   运行此文件，将读取 `logs/test_exp2/` 中的最新实验CSV，生成 Nature / IEEE 风格的矢量图谱（雷达图、热力图、盒须图、错误分布环形图等）。

## 7. 常见排障入口（For LLM/Developers）

- **动捕数据没法注入**？
  -> 看 `MocapClient.py` UDP端口对不对。看 `Communication_Mavlink` 的 `_send_vision_position()` 函数里的反射名是否匹配底层版本。
- **指令一直抛出“安全保护触发”**？
  -> 在 `Config.json` 检查 `alt_floor_ned` (-0.3等负数表示离地高度)。
- **“AI 生成执行”环节的 Python 代码异常退出**？
  -> 去 `ServerFile/logs/code/` 下看当时那条命令实际拼接下发的 `.py` 源文件。特别留意LLM有没有幻觉生成如 `import` 违规库的情况（沙箱已拦截常见风险）。

---
**最小上下文结语**：
如果你是AI助手，处理新需求时请始终牢记：**配置在Config，线程控制在Agent，动作与安全拦截在Communication。请勿破坏看门狗机制与底层的 Guard 拦截器。**

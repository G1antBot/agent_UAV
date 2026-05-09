'''
实验2场景初始化 - 标准测试环境
在 CameraRoom 中放置实验物体:
  copterID=2: 红色气球
  copterID=3: 蓝色小球
  copterID=4: 无人机模型（放在地上）
  copterID=100005: 小车（放在地上）

坐标系: NED (X=北, Y=东, Z=下, 负Z=上)
无柱子，所有物体在地面可见
'''

import UE4CtrlAPI
import time

ue = UE4CtrlAPI.UE4CtrlAPI()

# === 模型 ID ===
# 100000501: 红色气球
# 102000152: 蓝色小球
# 310:       无人机视觉模型
# 814:       小车

# === 放置红色气球（copterID=2）===
ue.sendUE4PosScale(copterID=2, vehicleType=100000501, PosE=[5, -1, -1.2])

# === 放置蓝色小球（copterID=3）===
ue.sendUE4PosScale(copterID=3, vehicleType=102000152, Scale=[0.05, 0.05, 0.05], PosE=[5, 2, -1.12])

# === 放置无人机模型（copterID=4）- 地面 ===
ue.sendUE4PosScale(copterID=4, vehicleType=310, PosE=[4, 0.5, 0])

# === 放置小车（copterID=100005）- 地面 ===
ue.sendUE4PosScale(copterID=100005, vehicleType=814, PosE=[6, -2, 0])
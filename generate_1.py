'''
实验3场景初始化 - 与 test_exp3.py 配套使用
在 CameraRoom 中预放置实验物体:
  copterID=2: 红色气球（目标），放在第一个试次的位置
  copterID=3: 红色气球（干扰），初始隐藏在地下
  copterID=4, 100005: 隐藏（无人机模型、小车）

坐标系: NED (X=北, Y=东, Z=下, 负Z=上)
HOME_POS = [-4, 0]  无人机边缘起始位置
'''

import UE4CtrlAPI
import time

ue = UE4CtrlAPI.UE4CtrlAPI()

# === 模型 ID ===
MODEL_RED_BALLOON = 100000501   # 红色气球
MODEL_DRONE_VIS   = 310         # 无人机视觉模型
MODEL_CAR         = 814         # 小车

# === 初始位置（与 test_exp3.py 一致）===
# 第一个试次: 距离2m, 工况A, scale=0.5
# 目标位置 = HOME(-4,0) + 2m向前 = (-2, 0, -0.3)
TARGET_INIT_POS   = [-2.0, 0.0, -0.3]
TARGET_INIT_SCALE = [0.5, 0.5, 0.5]
HIDDEN_POS        = [0, 0, 50]       # 地下50m，不可见

# === 放置红色气球（copterID=2）===
ue.sendUE4PosScale(copterID=2, vehicleType=MODEL_RED_BALLOON,
                   PosE=TARGET_INIT_POS, Scale=TARGET_INIT_SCALE)
print(f"[OK] 红色气球 -> pos={TARGET_INIT_POS} scale={TARGET_INIT_SCALE}")

# === 放置干扰球（copterID=3）- 初始隐藏 ===
ue.sendUE4PosScale(copterID=3, vehicleType=MODEL_RED_BALLOON,
                   PosE=HIDDEN_POS, Scale=[0.01, 0.01, 0.01])
print(f"[OK] 干扰球(隐藏) -> underground")

# === 隐藏无人机模型和小车 ===
ue.sendUE4PosScale(copterID=4, vehicleType=MODEL_DRONE_VIS,
                   PosE=HIDDEN_POS, Scale=[0.01, 0.01, 0.01])
ue.sendUE4PosScale(copterID=100005, vehicleType=MODEL_CAR,
                   PosE=HIDDEN_POS, Scale=[0.01, 0.01, 0.01])
print(f"[OK] 无人机模型+小车 -> underground")

print("\n场景初始化完成，可以运行 test_exp3.py")

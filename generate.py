'''
# 创建障碍物（柱子，创建，一定范围内随机）
# 控制小车行走，轨迹随机

H:810
Car:814

'''




import UE4CtrlAPI
import math
import time


ue = UE4CtrlAPI.UE4CtrlAPI()

# ue.sendUE4Cmd(cmd = "RflyChangeMapbyName MatchScene2025")

#ue.sendUE4PosScale(1,10100310)
#time.sleep(2)
# ue.sendUE4ExtAct(1,[1, -35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

#设置起飞标志
# ue.sendUE4PosScale(copterID=100001,vehicleType=810,PosE=[0,0,0])

# #设置静态柱子
ue.sendUE4PosScale(copterID=100002,vehicleType=824,PosE=[5,-1,2.0])
ue.sendUE4PosScale(copterID=100003,vehicleType=824,PosE=[5,2,2.0])
ue.sendUE4PosScale(copterID=100004,vehicleType=824,PosE=[4,0.5,2.0])



#以下代码调试使用Python 控小车

# ue.sendUE4PosScale(copterID=2,vehicleType=825,PosE=[7.17,0,0])
# ue.sendUE4PosScale(copterID=2,vehicleType=825,PosE=[5,-1,-1])
ue.sendUE4PosScale(copterID=2,vehicleType=100000501,PosE=[5,-1,-1.2])

ue.sendUE4PosScale(copterID=3,vehicleType=102000152,Scale=[0.05, 0.05, 0.05],PosE=[5,2,-1.12])

ue.sendUE4PosScale(copterID=4,vehicleType=310,PosE=[4,0.5,-1])


# time.sleep(5)
# ue.sendUE4ExtAct(2,[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
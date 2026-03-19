"""
坐标系转换函数: 使用 scipy 将机体坐标系位移转换为北东地（NED）坐标系位移。

参数:
    dx_body (float): 机体系X轴位移（机头方向为正）
    dy_body (float): 机体系Y轴位移（右侧为正）
    dz_body (float): 机体系Z轴位移（向下为正）
    roll (float): 滚转角（绕机体系X轴，弧度）
    pitch (float): 俯仰角（绕机体系Y轴，弧度）
    yaw (float): 偏航角（绕机体系Z轴，弧度）

返回:
    tuple[float, float, float]: NED坐标系下的位移 (dx_ned, dy_ned, dz_ned)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def body_to_ned(dx_body: float, dy_body: float, dz_body: float, roll: float, pitch: float, yaw: float) -> tuple[float, float, float]:
    rotation = R.from_euler('ZYX', [yaw, pitch, roll], degrees=False)  # 定义欧拉角顺序：Z（偏航）→ Y（俯仰）→ X（滚转）（符合无人机姿态定义）
    v_body = np.array([dx_body, dy_body, dz_body])                     # 机体系位移向量（列向量）
    v_ned = rotation.apply(v_body)                                     # 转换为NED系位移向量（旋转矩阵右乘向量）
    return v_ned[0], v_ned[1], v_ned[2]


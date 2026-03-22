# Mavlink通信文件
import cv2
import numpy as np
import time
import sys
sys.path.append(r"D:\Rflysim\RflySimAPIs\RflySimSDK\vision")
import VisionCaptureApi
import PX4MavCtrlV4 as PX4MavCtrl
import ReqCopterSim
from openai import OpenAI
import base64
import torch
from PIL import Image
from torchvision.ops import box_convert
from ultralytics import YOLOE

from PIL import Image

import math


class BodyCommMavlink(object):
    def __init__(self):
        # 检查是否使用GPU
        if torch.cuda.is_available():
            print("use_gpu")
            self.is_cup = False
        else:
            print("use_cpu")
            self.is_cup = True


        # 初始化火山引擎LLM客户端
        api_key = "24572520-5c64-4470-8c3d-5ecb84781725"
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3 ",
        )

        # 加载YOLOE模型
        self.yolo_model = YOLOE("i:/drone_project/实验6-7_无人机视觉语言控制实验/1.软件在环实验/ServerFile/weights/best.pt")
        self.CONF_THRESHOLD = 0.25  # 置信度阈值
        self.NMS_THRESHOLD = 0.45   # NMS阈值

        # 初始化ReqCopterSim实例，用于与模拟器通信
        self.req = ReqCopterSim.ReqCopterSim()
        StartCopterID = 1  # 起始无人机ID
        TargetIP = self.req.getSimIpID(StartCopterID)  # 获取目标模拟器的IP地址

        # 初始化VisionCaptureApi实例，用于获取前置摄像头的图像
        self.vis = VisionCaptureApi.VisionCaptureApi(TargetIP)
        self.vis.jsonLoad()  # 加载配置文件
        self.vis.sendReqToUE4(0, TargetIP)  # 向UE4发送请求
        self.vis.startImgCap()  # 启动图像捕获

        # 初始化无人机列表
        self.VehilceNum = 1  # 无人机数量
        self.MavList = []
        for i in range(self.VehilceNum):
            CopterID = StartCopterID + i  # 当前无人机ID
            TargetIP = self.req.getSimIpID(CopterID)  # 获取目标模拟器的IP地址
            self.req.sendReSimIP(CopterID)  # 将无人机连接到指定的模拟器IP地址
            time.sleep(1)
            self.MavList = self.MavList + [PX4MavCtrl.PX4MavCtrler(CopterID, TargetIP)]  # 创建无人机控制器实例并添加到列表中
        time.sleep(2)

        # 初始化Mavlink循环
        for i in range(self.VehilceNum):
            self.MavList[i].InitMavLoop()  # 初始化每架无人机的Mavlink循环
        time.sleep(2)

        # 计算全局坐标（UE4地图）与NED坐标（无人机本地）的偏移量
        self.Error2UE4Map = []
        for i in range(self.VehilceNum):
            mav = self.MavList[i]
            self.Error2UE4Map = self.Error2UE4Map + [
                -np.array([
                    mav.uavGlobalPos[0] - mav.uavPosNED[0],  # X轴偏移
                    mav.uavGlobalPos[1] - mav.uavPosNED[1],  # Y轴偏移
                    mav.uavGlobalPos[2] - mav.uavPosNED[2]  # Z轴偏移
                ])
            ]

    def GetBodyMavList(self):
        # 返回无人机列表、无人机数量和坐标偏移量
        return self.MavList, self.VehilceNum, self.Error2UE4Map

    def detect_yolo(self, object_names):
        # 使用YOLO模型进行目标检测
        image = self.vis.Img[0].copy()
        results = self.yolo_model.track(image, conf=self.CONF_THRESHOLD, save=False)

        if not results:
            print("[warn] 未获得推理结果")
            return [], [], [], None

        # 解析检测结果
        obj_list = [result.name for result in results]  # 检测到的目标名称
        obj_locs = [result.boxes.tolist() for result in results]  # 边界框坐标
        obj_logits = [result.conf for result in results]  # 置信度分数
        img_with_box = results[0].plot(masks=False)  # 带标注框的图像

        return obj_list, obj_locs, obj_logits, img_with_box

    def search_object(self, object_names):
        # 通过旋转无人机的偏航角搜索目标
        current_yaw = self.MavList[0].uavAngEular[2]
        for yaw_step in range(0, 360, 40):
            new_yaw = current_yaw + (yaw_step * 3.14159 / 180)  # 转换为弧度
            self.MavList[0].SendPosNED(
                self.MavList[0].uavPosNED[0],
                self.MavList[0].uavPosNED[1],
                self.MavList[0].uavPosNED[2],
                new_yaw
            )
            time.sleep(2)

            # 检测目标
            obj_list, obj_locs, obj_logits, img_with_box = self.detect_yolo(object_names)
            if object_names in obj_list:
                print(object_names, " found during rotation.")
                return True
        return False

    def cv2_to_base64(self, image, format='.png'):
        # 将OpenCV图像转换为Base64字符串
        success, buffer = cv2.imencode(format, image)
        if not success:
            raise ValueError("图片编码失败，请检查格式参数")
        img_bytes = buffer.tobytes()
        return base64.b64encode(img_bytes).decode('utf-8')

    def look(self):
        # 获取前置摄像头的图像，并通过火山引擎LLM进行图像理解
        rgb_image = self.vis.Img[0]
        base64_str = self.cv2_to_base64(rgb_image, ".png")
        response = self.llm_client.chat.completions.create(
            model="doubao-1-5-vision-pro-32k-250115",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "图片中有哪些目标，请给出名称即可，给出常见的，清晰可见的目标即可，多个目标名称之间用英文逗号分隔"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_str}"
                            }
                        },
                    ],
                }
            ],
            temperature=0.01
        )
        content = response.choices[0].message.content
        return content


    def approachObjective(self, error_x, error_y):
        """
        根据目标的误差控制无人机接近目标。
        :param error_x: 目标在X方向上的误差（像素值）。
        :param error_y: 目标在Y方向上的误差（像素值）。
        """
        # ---------------- 一次性初始化 ----------------
        if not hasattr(self, "_servo"):
            # 检查是否使用CPU
            is_cpu = bool(getattr(self, "is_cup", True))
            # 根据是否使用CPU，设置检测帧率
            det_fps = 0.4 if is_cpu else 10.0
            self._servo = {
                # —— 阶段控制 ——
                "phase": "YAW_ALIGN",  # 初始阶段：先对准偏航
                "yaw_hold_need": 3,  # 连续满足阈值的次数，用于确认偏航对准
                "yaw_hold_cnt": 0,  # 当前连续满足阈值的计数

                # —— 误差处理 ——
                "tau_err": 0.5 if is_cpu else 0.25,  # 低通滤波器的时间常数，用于平滑误差
                "db_x": 5.0, "db_y": 5.0,  # 死区（像素/角度），避免误差过小时的抖动
                "hit_x": 1.0, "hit_y": 1.0,  # 到达阈值，用于判断是否到达目标
                "hit_need": 3,  # 连续命中次数，用于确认到达目标

                # —— 偏航控制 ——
                "K_yaw": 0.0006,  # 偏航增益，控制偏航角速度的大小
                "yaw_max": math.radians(30),  # 最大偏航角速度（弧度/秒）
                "yaw_align_tol": 25.0,  # 认为“对准”的偏航误差阈值（像素/角度）

                # —— 速度合成（朝向目标） ——
                # 先 yaw 对准；对准后速度指向 X–Z 平面内“朝向目标”的方向
                # 用 ey -> alpha（俯仰方向角）来分解： vx=v*cos(alpha), vz=v*sin(alpha)
                "ay": 327.0,  # ey->alpha 的尺度，越小越敏感
                "alpha_max": math.radians(85),  # 最大俯仰方向角，防止直冲上下
                "v_nom": 0.5,  # 对准后朝向推进的标称速度
                "v_min": 0.05, "v_max": 1.0,  # 推进速度标量上下限
                "vz_max": 0.35,  # 垂直分量限幅（FRD：向下为正）

                # —— 安全与下发 ——
                "lost_timeout": max(3.0 / det_fps, 1.5),  # 目标丢失超时时间
                "hold_sec": max(0.8 / det_fps, 0.15),  # 指令保持时间

                # —— 运行态 ——
                "last_time": time.monotonic(),  # 上次运行时间
                "last_det_ts": time.monotonic(),  # 上次检测到目标的时间
                "lp_ex": 0.0, "lp_ey": 0.0,  # 低通滤波后的误差
                "hit_cnt": 0,  # 连续命中计数
                "last_cmd": (0.0, 0.0, 0.0, 0.0),  # 上次发送的指令
                "next_ok_ts": 0.0,  # 下次可以发送指令的时间
            }

        s = self._servo
        t = time.monotonic()  # 当前时间
        dt = t - s["last_time"];
        s["last_time"] = t  # 计算时间差并更新上次运行时间
        s["last_det_ts"] = t  # 更新上次检测到目标的时间

        # ---------------- 小工具 ----------------
        def clamp(v, vmin, vmax):
            """
            限制值v在[vmin, vmax]范围内。
            :param v: 输入值。
            :param vmin: 最小值。
            :param vmax: 最大值。
            :return: 限制后的值。
            """
            return vmin if v < vmin else (vmax if v > vmax else v)

        def lowpass(prev, cur, dt, tau):
            """
            一阶低通滤波器。
            :param prev: 上一次的值。
            :param cur: 当前的值。
            :param dt: 时间差。
            :param tau: 时间常数。
            :return: 滤波后的值。
            """
            a = dt / (tau + dt) if dt > 0 else 1.0
            return (1 - a) * prev + a * cur

        def deadband(e, db):
            """
            死区函数，当误差小于死区时，返回0，否则返回误差减去死区值。
            :param e: 误差。
            :param db: 死区值。
            :return: 处理后的误差。
            """
            return 0.0 if abs(e) <= db else (e - math.copysign(db, e))

        # ---------------- 误差预处理 ----------------
        s["lp_ex"] = lowpass(s["lp_ex"], error_x, dt, s["tau_err"])  # 低通滤波处理X方向误差
        s["lp_ey"] = lowpass(s["lp_ey"], error_y, dt, s["tau_err"])  # 低通滤波处理Y方向误差
        ex = deadband(s["lp_ex"], s["db_x"])  # 应用死区处理X方向误差
        ey = deadband(s["lp_ey"], s["db_y"])  # 应用死区处理Y方向误差

        # ---------------- 到达统计 ----------------
        if abs(s["lp_ex"]) <= s["hit_x"] and abs(s["lp_ey"]) <= s["hit_y"]:
            # 如果误差在到达阈值内，增加连续命中计数
            s["hit_cnt"] = min(s["hit_cnt"] + 1, s["hit_need"])
        else:
            # 否则重置连续命中计数
            s["hit_cnt"] = 0

        # ---------------- 丢失保护 ----------------
        if (t - s["last_det_ts"]) > s["lost_timeout"]:
            # 如果目标丢失超时，停止无人机
            cmd = (0.0, 0.0, 0.0, 0.0)
        else:
            # ========== 阶段 1：先 yaw 对准 ==========
            if s["phase"] == "YAW_ALIGN":
                # 计算偏航角速度
                yawrate = clamp(s["K_yaw"] * ex, -s["yaw_max"], s["yaw_max"])
                vx = 0.0  # 对准阶段不推进（也可以给很小的前进速度）
                vy = 0.0
                vz = 0.0

                # 判定是否对准：|ex| < 阈值 且 连续满足
                if abs(s["lp_ex"]) <= s["yaw_align_tol"]:
                    s["yaw_hold_cnt"] += 1
                else:
                    s["yaw_hold_cnt"] = 0

                if s["yaw_hold_cnt"] >= s["yaw_hold_need"]:
                    s["phase"] = "APPROACH"  # 转入推进阶段

                cmd = (vx, vy, vz, yawrate)

            # ========== 阶段 2：朝向目标推进 ==========
            else:  # "APPROACH"
                # 若偏航又变大，退回对准阶段
                if abs(s["lp_ex"]) > 1.5 * s["yaw_align_tol"]:
                    s["phase"] = "YAW_ALIGN"
                    # 立即给一次对准指令（可选）
                    yawrate = clamp(s["K_yaw"] * ex, -s["yaw_max"], s["yaw_max"])
                    cmd = (0.0, 0.0, 0.0, yawrate)
                else:
                    # 偏航微调
                    yawrate = clamp(s["K_yaw"] * ex, -s["yaw_max"], s["yaw_max"])

                    # 将 ey 映射为俯仰方向角 alpha（X–Z 平面方向）
                    alpha = math.atan(ey / s["ay"]) if s["ay"] != 0 else 0.0
                    alpha = clamp(alpha, -s["alpha_max"], s["alpha_max"])
                    # 推进速度标量（可按误差适度减小速度，先对准再推进）
                    # 简单做法：误差越大，速度越小
                    slow = 1.0 / (1.0 + (abs(ex)/ (3*s["yaw_align_tol"]))**2 + (abs(ey)/ s["ay"])**2)
                    v = clamp(s["v_nom"] * slow, s["v_min"], s["v_max"])
                    # 分解到 X（前）和 Z（下）方向：速度“朝向目标”
                    vx = v * math.cos(alpha)
                    vz = v * math.sin(alpha)
                    vz = clamp(vz, -s["vz_max"], s["vz_max"])
                    vy = 0.0

                    # 到达 → 悬停
                    if s["hit_cnt"] >= s["hit_need"]:
                        vx = vy = vz = yawrate = 0.0

                    cmd = (vx, vy, vz, yawrate)

            # ---------------- 末级下发（节流+轻微平滑） ----------------
            if t >= s["next_ok_ts"]:
                beta = dt/(0.15+dt) if dt > 0 else 1.0
                smooth_cmd = tuple((1-beta)*c_old + beta*c_new for c_old, c_new in zip(s["last_cmd"], cmd))
                # print("phase:", s["phase"], "cmd:", smooth_cmd)  # 如需调试可打开
                if smooth_cmd != s["last_cmd"]:
                    self.MavList[0].SendVelFRD(*smooth_cmd)
                    if self.is_cup:
                        #使用cpu的时候检测一帧图像要2s-3s，要一步一步的靠近气球
                        time.sleep(1.0)
                        self.MavList[0].SendVelFRD(0, 0, 0, 0)
                    s["last_cmd"] = smooth_cmd
                s["next_ok_ts"] = t + s["hold_sec"]

    def faceObjectiveOnly(self, error_x, error_y):
        # ---------------- 一次性初始化 ----------------
        if not hasattr(self, "_face_servo"):
            is_cpu = bool(getattr(self, "is_cup", True))
            det_fps = 0.4 if is_cpu else 10.0
            self._face_servo = {
                "phase": "YAW_ALIGN",
                "yaw_hold_need": 3,
                "yaw_hold_cnt": 0,

                "tau_err": 0.5 if is_cpu else 0.25,
                "db_x": 5.0,
                "yaw_align_tol": 25.0,
                "K_yaw": 0.0006,
                "yaw_max": math.radians(30),

                "lost_timeout": max(3.0 / det_fps, 1.5),
                "hold_sec": max(0.8 / det_fps, 0.15),
                "last_time": time.monotonic(),
                "last_det_ts": time.monotonic(),
                "lp_ex": 0.0,
                "last_cmd": (0.0, 0.0, 0.0, 0.0),
                "next_ok_ts": 0.0,
            }

        s = self._face_servo
        t = time.monotonic()
        dt = t - s["last_time"]
        s["last_time"] = t
        s["last_det_ts"] = t
        def clamp(v, vmin, vmax):
            return vmin if v < vmin else (vmax if v > vmax else v)

        def lowpass(prev, cur, dt, tau):
            a = dt / (tau + dt) if dt > 0 else 1.0
            return (1 - a) * prev + a * cur

        def deadband(e, db):
            return 0.0 if abs(e) <= db else (e - math.copysign(db, e))

        # 误差滤波
        s["lp_ex"] = lowpass(s["lp_ex"], error_x, dt, s["tau_err"])
        ex = deadband(s["lp_ex"], s["db_x"])

        # 丢失保护
        if t - s["last_det_ts"] > s["lost_timeout"]:
            cmd = (0.0, 0.0, 0.0, 0.0)
        else:
            # 【全程只做偏航对准，绝不前进】
            yawrate = clamp(s["K_yaw"] * ex, -s["yaw_max"], s["yaw_max"])
            cmd = (0.0, 0.0, 0.0, yawrate)  # 👈 速度全是 0，只转不飞

        # 下发指令
        if t >= s["next_ok_ts"]:
            beta = dt / (0.15 + dt) if dt > 0 else 1.0
            smooth_cmd = tuple((1 - beta) * c_old + beta * c_new for c_old, c_new in zip(s["last_cmd"], cmd))
            if smooth_cmd != s["last_cmd"]:
                self.MavList[0].SendVelFRD(*smooth_cmd)
                if self.is_cup:
                    time.sleep(1.0)
                    self.MavList[0].SendVelFRD(0, 0, 0, 0)
                s["last_cmd"] = smooth_cmd
            s["next_ok_ts"] = t + s["hold_sec"]

    def save_detection_image(self):
        """
        保存当前带有检测结果的摄像头图片。
        """
        # 调用检测函数，获取检测结果
        results = self.detect_yolo("")

        if results:
            # 获取带有检测框的图片
            img_with_box = results[3]
            if img_with_box is not None:
                # 保存图片
                img_with_box.save("current_detection.png")
                print("当前检测图片已保存为current_detection.png")
            else:
                print("未能生成带有检测框的图片。")
        else:
            print("未检测到任何结果，无法保存图片。")
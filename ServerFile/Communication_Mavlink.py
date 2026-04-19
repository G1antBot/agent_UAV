# Mavlink通信文件
import cv2
import numpy as np
import time
import threading
import sys
import os
import json
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
import re
import traceback
from datetime import datetime
from runtime_logger import get_runtime_logger
from Coordinate_Transformation import body_to_ned as b2n


class BodyCommMavlink(object):
    def __init__(self):
        self.logger = get_runtime_logger("comm")
        self._runtime_cfg = self._load_runtime_config()
        self.run_mode = self._load_run_mode_from_config()
        self._image_source_mode = "visioncapture" if self.run_mode == "sim" else "opencv"
        self._real_cam = None
        self._camera_lock = threading.Lock()
        self._detect_lock = threading.Lock()
        self._preview_cfg = self._load_realtime_preview_config()
        self._preview_running = False
        self._preview_thread = None
        self._preview_stop_event = threading.Event()
        self._preview_last_status = "未启动"
        self._real_camera_index = 0
        # 检查是否使用GPU
        if torch.cuda.is_available():
            print("use_gpu")
            self.is_cup = False
        else:
            print("use_cpu")
            self.is_cup = True
        self.logger.info(f"通信模块启动, run_mode={self.run_mode}, is_cpu={self.is_cup}")


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
        self.last_detection_image = None
        self.last_detection_time = None
        self.last_detection_has_object = False
        self.last_search_result_cn = "暂无搜索结果"
        self.logger.info("YOLOE模型加载完成")

        # 最低硬约束：仿真和实飞默认一致开启
        self._safety_cfg = {
            "enabled": True,
            "single_vehicle_only": True,
            "enable_space_fence": False,
            "max_radius_m": 3.0,
            "task_timeout_s": 180.0,
            "timeout_action": "hover",
            "projection_dt_s": 0.35,
            "alt_ned_min": -1.8,
            "alt_ned_max": -0.3,
            "motion_limits": {
                "generic": {"xy": 0.8, "z": 0.35, "yawrate_deg": 45.0},
                "search": {"xy": 0.8, "z": 0.35, "yawrate_deg": 45.0},
                "face": {"xy": 0.0, "z": 0.0, "yawrate_deg": 45.0},
                "approach": {"xy": 1.0, "z": 0.35, "yawrate_deg": 45.0},
                "strike": {"xy": 1.2, "z": 0.35, "yawrate_deg": 60.0},
                "land": {"xy": 0.0, "z": 0.0, "yawrate_deg": 0.0},
            },
        }
        self._safety_motion_mode = "generic"
        # 仅在任务开始时由上层显式设置，避免把程序空闲时间计入任务超时。
        self._safety_task_start_ts = None
        self._safety_guards_installed = False
        self._mocap_bridge_running = False
        self._mocap_bridge_thread = None
        self._mocap_bridge_stop_event = threading.Event()
        self._mocap_pose_provider = None
        # 动捕坐标默认按ENU解释并映射到NED，可按实验室坐标系调整
        self._mocap_cfg = {
            "x_axis": "x",   # mocap x -> north
            "y_axis": "y",   # mocap y -> east
            "z_axis": "z",   # mocap z -> up(随后转NED)
            "x_sign": 1.0,
            "y_sign": 1.0,
            "z_sign": 1.0,
            "pos_offset_ned": [0.0, 0.0, 0.0],
            "rpy_offset_rad": [0.0, 0.0, 0.0],
        }

        # 运行模式初始化：sim沿用RflySim链路，real_mocap不再写死仿真图像链路
        self.req = None
        self.vis = None
        StartCopterID = 1  # 起始无人机ID
        if self.run_mode == "sim":
            self.req = ReqCopterSim.ReqCopterSim()
            TargetIP = self.req.getSimIpID(StartCopterID)

            self.vis = VisionCaptureApi.VisionCaptureApi(TargetIP)
            self.vis.jsonLoad()
            self.vis.sendReqToUE4(0, TargetIP)
            self.vis.startImgCap()
        else:
            TargetIP = "127.0.0.1"
            self._init_real_camera_from_config()
            self.logger.warning("run_mode=real_mocap: 已启用真实相机取流与动捕桥接模式")

        # 初始化无人机列表
        self.VehilceNum = 1  # 无人机数量
        self.MavList = []
        for i in range(self.VehilceNum):
            CopterID = StartCopterID + i  # 当前无人机ID
            if self.run_mode == "sim" and self.req is not None:
                TargetIP = self.req.getSimIpID(CopterID)
                self.req.sendReSimIP(CopterID)
            else:
                TargetIP = "127.0.0.1"
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

        self._home_pos_ned = np.array([
            float(self.MavList[0].uavPosNED[0]),
            float(self.MavList[0].uavPosNED[1]),
            float(self.MavList[0].uavPosNED[2]),
        ])
        self._install_motion_safety_guards()
        self.logger.info(
            f"安全约束已启用: radius={self._safety_cfg['max_radius_m']}m, timeout={self._safety_cfg['task_timeout_s']}s, "
            f"alt_ned=[{self._safety_cfg['alt_ned_min']},{self._safety_cfg['alt_ned_max']}], "
            f"space_fence={self._safety_cfg['enable_space_fence']}, single_vehicle_only={self._safety_cfg['single_vehicle_only']}"
        )

    def _load_runtime_config(self):
        """加载运行配置（兼容Config.json中的注释）。"""
        config_path = os.path.join(os.path.dirname(__file__), "Config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw = f.read()
            cleaned = re.sub(r"//.*", "", raw)
            return json.loads(cleaned)
        except Exception as e:
            self.logger.warning(f"读取Config.json失败，使用默认配置: {e}")
            return {}

    def _load_run_mode_from_config(self):
        """从Config.json读取运行模式，默认sim。"""
        try:
            cfg = self._runtime_cfg if isinstance(self._runtime_cfg, dict) else {}
            mode = str(cfg.get("run_mode", "sim")).strip().lower()
            if mode in ("sim", "real_mocap"):
                return mode
            self.logger.warning(f"Config.json run_mode非法({mode})，回退sim")
            return "sim"
        except Exception as e:
            self.logger.warning(f"解析run_mode失败，回退sim: {e}")
            return "sim"

    def is_mock_mocap_allowed(self):
        """是否允许real_mocap模式下自动注入mock动捕回调（默认False）。"""
        cfg = self._runtime_cfg if isinstance(self._runtime_cfg, dict) else {}
        raw = cfg.get("allow_mock_mocap_for_debug", False)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(raw, (int, float)):
            return bool(raw)
        return False

    def _load_realtime_preview_config(self):
        """读取实飞实时预览配置。"""
        cfg = self._runtime_cfg if isinstance(self._runtime_cfg, dict) else {}
        cam_cfg = cfg.get("real_camera", {}) if isinstance(cfg.get("real_camera", {}), dict) else {}
        preview = cam_cfg.get("preview", {}) if isinstance(cam_cfg.get("preview", {}), dict) else {}

        def _as_bool(v, default):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "on")
            if isinstance(v, (int, float)):
                return bool(v)
            return default

        def _as_float(v, default, low=None, high=None):
            try:
                val = float(v)
            except Exception:
                val = float(default)
            if low is not None:
                val = max(low, val)
            if high is not None:
                val = min(high, val)
            return val

        return {
            "auto_start": _as_bool(preview.get("auto_start", True), True),
            "window_name": str(preview.get("window_name", "RealCam YOLO Preview")).strip() or "RealCam YOLO Preview",
            "detect_hz": _as_float(preview.get("detect_hz", 8.0), 8.0, low=0.5, high=60.0),
            "target_name": str(preview.get("target_name", "")).strip(),
            "show_overlay": _as_bool(preview.get("show_overlay", True), True),
            "hotkey_save": str(preview.get("hotkey_save", "s")).strip().lower()[:1] or "s",
            "hotkey_quit": str(preview.get("hotkey_quit", "q")).strip().lower()[:1] or "q",
        }

    def _init_real_camera_from_config(self):
        """实飞模式初始化真实相机。"""
        cfg = self._runtime_cfg if isinstance(self._runtime_cfg, dict) else {}
        cam_cfg = cfg.get("real_camera", {}) if isinstance(cfg.get("real_camera", {}), dict) else {}
        source = str(cam_cfg.get("source", "opencv")).strip().lower()
        self._image_source_mode = source

        if source != "opencv":
            self.logger.warning(f"暂不支持的real_camera.source={source}，回退opencv")
            self._image_source_mode = "opencv"

        cam_index = int(cam_cfg.get("device_index", 0))
        self._real_camera_index = cam_index
        width = int(cam_cfg.get("width", 640))
        height = int(cam_cfg.get("height", 480))
        fps = int(cam_cfg.get("fps", 30))

        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            self.logger.warning(f"真实相机打开失败: index={cam_index}")
            self._real_cam = None
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        self._real_cam = cap
        self.logger.info(f"真实相机初始化完成: index={cam_index}, size={width}x{height}, fps={fps}")

    def _read_real_camera_frame(self):
        """从真实相机读取一帧。"""
        if self._real_cam is None:
            raise RuntimeError("真实相机未初始化")
        with self._camera_lock:
            ok, frame = self._real_cam.read()
        if not ok or frame is None:
            raise RuntimeError("真实相机读帧失败")
        return frame.copy()

    def _get_current_frame(self):
        """统一取帧入口：sim读取VisionCaptureApi，real_mocap读取真实相机。"""
        if self.run_mode == "sim":
            if self.vis is None or not hasattr(self.vis, "Img") or len(getattr(self.vis, "Img", [])) == 0:
                raise RuntimeError("sim模式图像源不可用，请检查VisionCaptureApi")
            frame = self.vis.Img[0]
            if frame is None:
                raise RuntimeError("sim模式图像源尚未就绪")
            return frame.copy()

        if self._image_source_mode == "opencv":
            return self._read_real_camera_frame()

        raise RuntimeError(f"不支持的图像源模式: {self._image_source_mode}")

    def _install_motion_safety_guards(self):
        """在飞控对象上安装统一安全出口，拦截速度/位置/降落指令。"""
        if self._safety_guards_installed:
            return
        if not self.MavList:
            return

        mav = self.MavList[0]
        original_send_vel_frd = getattr(mav, "SendVelFRD", None)
        original_send_pos_ned = getattr(mav, "SendPosNED", None)
        original_send_pos_ned_no_yaw = getattr(mav, "SendPosNEDNoYaw", None)
        original_send_pos_ned_ext = getattr(mav, "SendPosNEDExt", None)
        original_send_land = getattr(mav, "sendMavLand", None)

        if not callable(original_send_vel_frd) or not callable(original_send_pos_ned):
            self.logger.warning("飞控对象缺少必要控制接口，安全出口未安装")
            return

        def _clamp(value, low, high):
            return low if value < low else (high if value > high else value)

        def _current_mode():
            return (getattr(self, "_safety_motion_mode", "generic") or "generic").lower()

        def _current_limits():
            mode = _current_mode()
            limits = self._safety_cfg.get("motion_limits", {})
            return limits.get(mode, limits.get("generic", {"xy": 0.8, "z": 0.35, "yawrate_deg": 45.0}))

        def _warn(reason, detail=""):
            msg = f"[SAFETY] {reason}"
            if detail:
                msg = f"{msg}: {detail}"
            self.logger.warning(msg)

        def _timeout_active():
            start_ts = getattr(self, "_safety_task_start_ts", None)
            if start_ts is None:
                return False
            return (time.monotonic() - float(start_ts)) > float(self._safety_cfg.get("task_timeout_s", 60.0))

        def _send_hover():
            return original_send_vel_frd(0.0, 0.0, 0.0, 0.0)

        def _project_after_velocity(vx, vy, vz):
            roll, pitch, yaw = mav.uavAngEular[0], mav.uavAngEular[1], mav.uavAngEular[2]
            dt = float(self._safety_cfg.get("projection_dt_s", 0.35))
            dx_ned, dy_ned, dz_ned = b2n(vx * dt, vy * dt, vz * dt, roll, pitch, yaw)
            cur_x, cur_y, cur_z = float(mav.uavPosNED[0]), float(mav.uavPosNED[1]), float(mav.uavPosNED[2])
            return np.array([cur_x + dx_ned, cur_y + dy_ned, cur_z + dz_ned], dtype=float)

        def _check_fence(projected_pos, label):
            if not bool(self._safety_cfg.get("enable_space_fence", True)):
                return True
            home = getattr(self, "_home_pos_ned", None)
            if home is None:
                return True
            radius = float(np.linalg.norm(projected_pos[:2] - home[:2]))
            alt_ned = float(projected_pos[2])
            if radius > float(self._safety_cfg.get("max_radius_m", 3.0)):
                _warn("空间围栏触发", f"{label} radius={radius:.2f}m > {self._safety_cfg['max_radius_m']:.2f}m")
                _send_hover()
                return False
            if alt_ned < float(self._safety_cfg.get("alt_ned_min", -1.8)) or alt_ned > float(self._safety_cfg.get("alt_ned_max", -0.3)):
                _warn("高度围栏触发", f"{label} alt_ned={alt_ned:.2f} outside [{self._safety_cfg['alt_ned_min']:.2f},{self._safety_cfg['alt_ned_max']:.2f}]")
                _send_hover()
                return False
            return True

        def guarded_send_vel_frd(vx=0, vy=0, vz=0, yawrate=0):
            if not bool(self._safety_cfg.get("enabled", True)):
                return original_send_vel_frd(vx, vy, vz, yawrate)

            if _timeout_active():
                _warn("任务超时", f"mode={_current_mode()} -> 悬停")
                return _send_hover()

            limits = _current_limits()
            xy_limit = float(limits.get("xy", 0.8))
            z_limit = float(limits.get("z", 0.35))
            yaw_limit = math.radians(float(limits.get("yawrate_deg", 45.0)))

            clamped_vx = _clamp(float(vx), -xy_limit, xy_limit)
            clamped_vy = _clamp(float(vy), -xy_limit, xy_limit)
            clamped_vz = _clamp(float(vz), -z_limit, z_limit)
            clamped_yawrate = _clamp(float(yawrate), -yaw_limit, yaw_limit)
            if (
                clamped_vx != float(vx)
                or clamped_vy != float(vy)
                or clamped_vz != float(vz)
                or clamped_yawrate != float(yawrate)
            ):
                _warn(
                    "速度限幅触发",
                    f"mode={_current_mode()} cmd=({vx:.3f},{vy:.3f},{vz:.3f},{yawrate:.3f}) -> ({clamped_vx:.3f},{clamped_vy:.3f},{clamped_vz:.3f},{clamped_yawrate:.3f})"
                )

            projected = _project_after_velocity(clamped_vx, clamped_vy, clamped_vz)
            if not _check_fence(projected, f"mode={_current_mode()}"):
                return False
            return original_send_vel_frd(clamped_vx, clamped_vy, clamped_vz, clamped_yawrate)

        def guarded_send_pos_ned(x=math.nan, y=math.nan, z=math.nan, yaw=math.nan):
            if not bool(self._safety_cfg.get("enabled", True)):
                return original_send_pos_ned(x, y, z, yaw)

            if _timeout_active():
                _warn("任务超时", f"mode={_current_mode()} -> 悬停")
                return _send_hover()

            try:
                target = np.array([float(x), float(y), float(z)], dtype=float)
            except Exception:
                _warn("位置指令参数非法", f"mode={_current_mode()} target=({x},{y},{z})")
                return _send_hover()

            if not _check_fence(target, f"mode={_current_mode()}"):
                return False
            return original_send_pos_ned(x, y, z, yaw)

        def guarded_send_pos_ned_no_yaw(x=math.nan, y=math.nan, z=math.nan):
            cur_yaw = float(mav.uavAngEular[2])
            return guarded_send_pos_ned(x, y, z, cur_yaw)

        def guarded_send_pos_ned_ext(x=math.nan, y=math.nan, z=math.nan, mode=3, isNED=True):
            cur_yaw = float(mav.uavAngEular[2])
            return guarded_send_pos_ned(x, y, z, cur_yaw)

        def guarded_send_land(xM, yM, zM):
            if not bool(self._safety_cfg.get("enabled", True)):
                return original_send_land(xM, yM, zM)
            if _timeout_active():
                _warn("任务超时", "land被降级为悬停")
                return _send_hover()
            return original_send_land(xM, yM, zM)

        mav.SendVelFRD = guarded_send_vel_frd
        mav.SendPosNED = guarded_send_pos_ned
        if callable(original_send_pos_ned_no_yaw):
            mav.SendPosNEDNoYaw = guarded_send_pos_ned_no_yaw
        if callable(original_send_pos_ned_ext):
            mav.SendPosNEDExt = guarded_send_pos_ned_ext
        if callable(original_send_land):
            mav.sendMavLand = guarded_send_land

        self._safety_guards_installed = True

    def get_safety_summary(self):
        limits = self._safety_cfg.get("motion_limits", {})
        return {
            "enabled": bool(self._safety_cfg.get("enabled", True)),
            "enable_space_fence": bool(self._safety_cfg.get("enable_space_fence", True)),
            "radius_m": float(self._safety_cfg.get("max_radius_m", 3.0)),
            "timeout_s": float(self._safety_cfg.get("task_timeout_s", 60.0)),
            "alt_ned": [float(self._safety_cfg.get("alt_ned_min", -1.8)), float(self._safety_cfg.get("alt_ned_max", -0.3))],
            "limits": limits,
            "single_vehicle_only": bool(self._safety_cfg.get("single_vehicle_only", True)),
        }

    def set_safety_motion_mode(self, mode: str):
        """设置当前动作模式，用于选择不同的速度上限。"""
        self._safety_motion_mode = (mode or "generic").lower()

    def set_task_start_timestamp(self, start_ts=None):
        """标记当前任务起点，用于统一60秒任务超时。"""
        self._safety_task_start_ts = float(start_ts if start_ts is not None else time.monotonic())

    def set_mocap_pose_provider(self, provider):
        """注入动捕位姿回调: provider() -> (x,y,z,roll,pitch,yaw)。"""
        self._mocap_pose_provider = provider

    def mocap_to_ned_pose(self, pose):
        """动捕坐标转NED: 输入(x,y,z,roll,pitch,yaw)，输出(n,e,d,roll,pitch,yaw)。"""
        if pose is None or len(pose) < 6:
            raise ValueError("mocap pose长度不足，需至少6个元素")

        x_raw, y_raw, z_raw, roll, pitch, yaw = [float(v) for v in pose[:6]]
        axis_map = {"x": x_raw, "y": y_raw, "z": z_raw}
        cfg = self._mocap_cfg

        north = float(cfg["x_sign"]) * axis_map[str(cfg["x_axis"]).lower()]
        east = float(cfg["y_sign"]) * axis_map[str(cfg["y_axis"]).lower()]
        up = float(cfg["z_sign"]) * axis_map[str(cfg["z_axis"]).lower()]

        # ENU/实验室up转NED的down
        down = -up

        pos_offset = cfg.get("pos_offset_ned", [0.0, 0.0, 0.0])
        n = north + float(pos_offset[0])
        e = east + float(pos_offset[1])
        d = down + float(pos_offset[2])

        rpy_offset = cfg.get("rpy_offset_rad", [0.0, 0.0, 0.0])
        roll = roll + float(rpy_offset[0])
        pitch = pitch + float(rpy_offset[1])
        yaw = yaw + float(rpy_offset[2])

        return n, e, d, roll, pitch, yaw

    def preflight_check(self):
        """启动预检：链路/状态检查，返回结构化结果。"""
        checks = []

        has_mav = bool(self.MavList) and callable(getattr(self.MavList[0], "InitMavLoop", None))
        checks.append({"name": "mavlink", "ok": has_mav, "detail": "Mavlink控制对象可用" if has_mav else "Mavlink控制对象不可用"})

        single_ok = (self.VehilceNum == 1)
        checks.append({"name": "single_vehicle", "ok": single_ok, "detail": f"VehilceNum={self.VehilceNum}"})

        state_ok = False
        if has_mav:
            try:
                pos = np.array(self.MavList[0].uavPosNED, dtype=float)
                ang = np.array(self.MavList[0].uavAngEular, dtype=float)
                state_ok = np.isfinite(pos).all() and np.isfinite(ang).all()
            except Exception:
                state_ok = False
        checks.append({"name": "state", "ok": state_ok, "detail": "位姿状态可读" if state_ok else "位姿状态不可读"})

        if self.run_mode == "sim":
            img_ok = False
            try:
                _ = self._get_current_frame()
                img_ok = True
            except Exception:
                img_ok = False
            checks.append({"name": "image_source", "ok": img_ok, "detail": "仿真图像链路可用" if img_ok else "仿真图像链路不可用"})
        else:
            provider_ok = callable(self._mocap_pose_provider)
            checks.append({"name": "mocap_provider", "ok": provider_ok, "detail": "已注入动捕位姿回调" if provider_ok else "未注入动捕位姿回调"})

            # real_mocap预检补充：桥接链路可用性（首包可转换且可注入飞控）
            bridge_ready_ok = False
            bridge_detail = "未执行桥接首包检查"
            if provider_ok and has_mav:
                try:
                    pose = self._mocap_pose_provider()
                    if not pose or len(pose) < 6:
                        bridge_detail = "动捕位姿回调首包无效"
                    else:
                        n, e, d, roll, pitch, yaw = self.mocap_to_ned_pose(pose)
                        sent = self._send_vision_position(n, e, d, roll, pitch, yaw)
                        if sent:
                            bridge_ready_ok = True
                            bridge_detail = "桥接链路可用(首包发送成功)"
                        else:
                            bridge_detail = "桥接链路不可用: 未找到外部位姿注入接口"
                except Exception as e:
                    bridge_detail = f"桥接链路检查异常: {e}"
            elif not provider_ok:
                bridge_detail = "桥接链路不可用: 未注入动捕位姿回调"
            elif not has_mav:
                bridge_detail = "桥接链路不可用: Mavlink控制对象不可用"

            checks.append({"name": "mocap_bridge", "ok": bridge_ready_ok, "detail": bridge_detail})

        ok = all(item["ok"] for item in checks)
        result = {
            "ok": ok,
            "run_mode": self.run_mode,
            "checks": checks,
        }
        self.logger.info(f"预检结果: {result}")
        return result

    def _send_vision_position(self, x, y, z, roll, pitch, yaw):
        """兼容不同飞控封装的外部位姿注入接口。"""
        if not self.MavList:
            return False
        mav = self.MavList[0]
        # 优先你要求的SendVisionPosition语义，同时兼容常见命名/参数签名
        for name in ["SendVisionPosition", "sendVisionPosition", "vision_position_estimate_send"]:
            fn = getattr(mav, name, None)
            if not callable(fn):
                continue
            try:
                fn(x, y, z, roll, pitch, yaw)
                return True
            except TypeError:
                try:
                    # 尝试 RflySim PX4MavCtrlV4 常用的 4 参数签名 (x, y, z, yaw)
                    fn(x, y, z, yaw)
                    return True
                except TypeError:
                    try:
                        # 尝试带时间戳的 7 参数签名
                        usec = int(time.time() * 1_000_000)
                        fn(usec, x, y, z, roll, pitch, yaw)
                        return True
                    except Exception:
                        continue
            except Exception:
                continue
        return False

    def start_mocap_bridge(self, hz=30.0):
        """启动动捕位姿桥接线程（仅real_mocap模式）。"""
        if self.run_mode != "real_mocap":
            self.logger.info("start_mocap_bridge跳过: 当前非real_mocap模式")
            return True
        if self._mocap_bridge_running:
            self.logger.info("动捕桥接线程已在运行")
            return True
        if not callable(self._mocap_pose_provider):
            self.logger.warning("启动动捕桥接失败: 未注入动捕位姿回调")
            return False

        # 启动前做一次首包硬校验：确保位姿可读且存在外部位姿注入接口。
        try:
            pose = self._mocap_pose_provider()
            if not pose or len(pose) < 6:
                self.logger.warning("启动动捕桥接失败: 动捕位姿回调返回无效首包")
                return False
            n, e, d, roll, pitch, yaw = self.mocap_to_ned_pose(pose)
            sent = self._send_vision_position(n, e, d, roll, pitch, yaw)
            if not sent:
                self.logger.warning("启动动捕桥接失败: 未找到外部位姿注入接口")
                return False
            self.logger.info("动捕桥接首包校验通过")
        except Exception as e:
            self.logger.warning(f"启动动捕桥接失败: 首包校验异常: {e}")
            return False

        period = max(1.0 / float(hz), 0.01)
        self._mocap_bridge_stop_event.clear()

        def _worker():
            self.logger.info(f"动捕桥接线程启动: hz={hz}")
            try:
                while not self._mocap_bridge_stop_event.is_set():
                    try:
                        pose = self._mocap_pose_provider()
                        if not pose or len(pose) < 6:
                            time.sleep(period)
                            continue
                        n, e, d, roll, pitch, yaw = self.mocap_to_ned_pose(pose)
                        sent = self._send_vision_position(n, e, d, roll, pitch, yaw)
                        if not sent:
                            self.logger.warning("动捕桥接发送失败: 未找到外部位姿注入接口")
                            break
                    except Exception as e:
                        self.logger.warning(f"动捕桥接异常: {e}")
                    time.sleep(period)
            finally:
                self._mocap_bridge_running = False
                self.logger.info("动捕桥接线程已退出")

        self._mocap_bridge_thread = threading.Thread(target=_worker, name="mocap_bridge", daemon=True)
        self._mocap_bridge_running = True
        self._mocap_bridge_thread.start()
        return True

    def stop_mocap_bridge(self):
        """停止动捕位姿桥接线程。"""
        if not self._mocap_bridge_running:
            return
        self._mocap_bridge_stop_event.set()
        if self._mocap_bridge_thread is not None:
            self._mocap_bridge_thread.join(timeout=1.5)
        self._mocap_bridge_thread = None
        self._mocap_bridge_running = False

    def _run_yolo_on_frame(self, frame_bgr, object_names=""):
        """对给定BGR帧执行一次YOLO检测并返回统一结果。"""
        start_ts = time.time()
        with self._detect_lock:
            results = self.yolo_model.track(frame_bgr, conf=self.CONF_THRESHOLD, save=False, verbose=False)

        if not results:
            img_raw = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            return [], [], [], img_raw, frame_bgr.copy(), (time.time() - start_ts) * 1000.0

        result = results[0]
        boxes = result.boxes
        names = result.names if hasattr(result, "names") else {}
        obj_list, obj_locs, obj_logits = [], [], []
        target_name = self._canonical_object_name(object_names)
        target_compact = self._compact_label(target_name) if target_name else ""

        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist() if boxes.cls is not None else []
            confs = boxes.conf.detach().cpu().numpy().tolist() if boxes.conf is not None else []
            locs = boxes.xyxy.detach().cpu().numpy().tolist() if boxes.xyxy is not None else []

            for idx, cls_id in enumerate(cls_ids):
                if isinstance(names, dict):
                    obj_name = names.get(cls_id, str(cls_id))
                elif isinstance(names, list) and 0 <= cls_id < len(names):
                    obj_name = names[cls_id]
                else:
                    obj_name = str(cls_id)
                canonical_obj_name = self._canonical_object_name(obj_name)
                obj_compact = self._compact_label(canonical_obj_name)
                if target_name and canonical_obj_name != target_name and obj_compact != target_compact:
                    continue
                obj_list.append(obj_name)
                obj_locs.append(locs[idx] if idx < len(locs) else [])
                obj_logits.append(float(confs[idx]) if idx < len(confs) else 0.0)

        plot_bgr = result.plot(masks=False)
        img_with_box = Image.fromarray(cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2RGB))
        return obj_list, obj_locs, obj_logits, img_with_box, plot_bgr, (time.time() - start_ts) * 1000.0

    def _draw_preview_overlay(self, frame_bgr, detect_ms, matched_count, status_text):
        if not bool(self._preview_cfg.get("show_overlay", True)):
            return frame_bgr
        view = frame_bgr.copy()
        target_name = self._preview_cfg.get("target_name", "") or "all"
        lines = [
            f"mode={self.run_mode} cam={self._real_camera_index}",
            f"target={target_name} matched={matched_count}",
            f"detect={detect_ms:.1f}ms ({self._preview_cfg.get('detect_hz', 8.0):.1f}Hz)",
            f"status={status_text}",
            "hotkeys: q=quit preview, s=save image",
        ]
        y = 24
        for line in lines:
            cv2.putText(view, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 255, 40), 2, cv2.LINE_AA)
            y += 24
        return view

    def start_realtime_preview(self):
        """启动实飞实时预览线程（带YOLO检测叠加）。"""
        if self._preview_running:
            self.logger.info("实时预览已在运行")
            return True
        if self.run_mode != "real_mocap":
            self.logger.info("start_realtime_preview跳过: 当前非real_mocap模式")
            return True
        if self._real_cam is None:
            self.logger.warning("启动实时预览失败: 真实相机未初始化")
            return False

        detect_hz = float(self._preview_cfg.get("detect_hz", 8.0))
        period = max(1.0 / detect_hz, 0.01)
        target_name = self._preview_cfg.get("target_name", "")
        window_name = self._preview_cfg.get("window_name", "RealCam YOLO Preview")
        key_quit = ord(str(self._preview_cfg.get("hotkey_quit", "q"))[0].lower())
        key_save = ord(str(self._preview_cfg.get("hotkey_save", "s"))[0].lower())

        self._preview_stop_event.clear()
        self._preview_last_status = "运行中"

        def _worker():
            self.logger.info(
                f"实时预览线程启动: window={window_name}, detect_hz={detect_hz:.1f}, target={target_name or 'all'}"
            )
            while not self._preview_stop_event.is_set():
                loop_start = time.time()
                try:
                    frame = self._read_real_camera_frame()
                    obj_list, obj_locs, obj_logits, img_with_box, plot_bgr, cost_ms = self._run_yolo_on_frame(frame, target_name)
                    self.last_detection_image = img_with_box
                    self.last_detection_time = datetime.now()
                    self.last_detection_has_object = len(obj_list) > 0
                    self._preview_last_status = "检测正常" if obj_list else "未检测到目标"

                    vis = self._draw_preview_overlay(
                        plot_bgr,
                        detect_ms=cost_ms,
                        matched_count=len(obj_list),
                        status_text=self._preview_last_status,
                    )
                    cv2.imshow(window_name, vis)

                    key = cv2.waitKey(1) & 0xFF
                    if key == key_quit:
                        self._preview_stop_event.set()
                        self._preview_last_status = "用户关闭预览"
                        self.logger.info("实时预览窗口收到退出按键")
                        break
                    if key == key_save:
                        try:
                            save_path = self.save_detection_image(use_latest=True)
                            self.logger.info(f"实时预览热键保存图片: {save_path}")
                        except Exception as e:
                            self.logger.warning(f"实时预览热键保存失败: {e}")
                except Exception as e:
                    self._preview_last_status = f"预览异常: {e}"
                    self.logger.warning(f"实时预览循环异常(已忽略，不影响飞控): {e}")

                elapsed = time.time() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)

            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass
            self.logger.info("实时预览线程退出")

        self._preview_thread = threading.Thread(target=_worker, name="realcam_preview", daemon=True)
        self._preview_thread.start()
        self._preview_running = True
        return True

    def stop_realtime_preview(self):
        """停止实飞实时预览线程。"""
        if not self._preview_running:
            return
        self._preview_stop_event.set()
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=1.5)
        self._preview_thread = None
        self._preview_running = False
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def close_image_source(self):
        """释放图像源资源。"""
        self.stop_realtime_preview()
        if self._real_cam is not None:
            try:
                self._real_cam.release()
                self.logger.info("真实相机已释放")
            except Exception:
                pass
            self._real_cam = None

    def _canonical_object_name(self, object_name):
        """
        将常见中文目标名归一化到检测模型使用的英文类别名。
        """
        if object_name is None:
            return ""
        name = self._normalize_label(object_name)
        if not name:
            return ""

        alias_map = {
            "蓝色小球": "blue ball",
            "蓝球": "blue ball",
            "蓝色球": "blue ball",
            "红色气球": "red balloon",
            "红气球": "red balloon",
            "红色球": "red balloon",
            "无人机": "uav",
            "无人飞机": "uav",
            "无人机目标": "uav",
            "drone": "uav",
            "drones": "uav",
            "uav": "uav",
            "quadcopter": "uav",
            "飞机": "airplane",
            "airplane": "airplane",
            "小车":"car",
            "小车":"car",
            "车":"car",
            "无人车":"car",
        }
        return alias_map.get(name, name)

    @staticmethod
    def _normalize_label(name):
        """
        统一标签文本：小写、去两端空白、下划线/连字符转空格、压缩多余空格。
        例如：blue_ball / blue-ball /  blue   ball  -> blue ball
        """
        text = str(name).strip().lower()
        text = re.sub(r"[_-]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _compact_label(name):
        """
        生成紧凑标签用于兜底匹配：去除空格、下划线、连字符。
        例如：blue_ball / blue ball / blue-ball -> blueball
        """
        text = str(name).strip().lower()
        return re.sub(r"[\s_-]+", "", text)

    @staticmethod
    def _wrap_angle_rad(angle_rad):
        """
        将角度归一化到[-pi, pi]。
        """
        a = float(angle_rad)
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    @staticmethod
    def _format_signed_deg(angle_deg):
        """
        将角度格式化为中文左右方向文本。
        """
        val = float(angle_deg)
        if abs(val) < 0.5:
            return "0°"
        if val > 0:
            return f"右{abs(val):.1f}°"
        return f"左{abs(val):.1f}°"

    @staticmethod
    def _cluster_angles_deg(angles_deg, threshold_deg=15.0):
        """
        对角度做一维聚类去重，避免旋转搜索时同一目标重复计数。
        """
        if not angles_deg:
            return []
        sorted_vals = sorted(float(v) for v in angles_deg)
        clusters = [[sorted_vals[0]]]
        for v in sorted_vals[1:]:
            if abs(v - clusters[-1][-1]) <= threshold_deg:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [sum(c) / len(c) for c in clusters]

    def _search_summary_line(self, canonical_name, mode, found, angle_list_deg):
        """
        生成一行中文搜索总结。
        """
        mode_cn = "快速搜索" if mode == "quick" else "全景搜索"
        if not found:
            return f"{mode_cn}: 未发现{canonical_name}"
        angles_text = ", ".join(self._format_signed_deg(v) for v in angle_list_deg)
        return f"{mode_cn}: 发现{canonical_name}{len(angle_list_deg)}个, 相对朝向[{angles_text}]"

    def GetBodyMavList(self):
        # 返回无人机列表、无人机数量和坐标偏移量
        return self.MavList, self.VehilceNum, self.Error2UE4Map

    def detect_yolo(self, object_names):
        # 使用YOLO模型进行目标检测
        try:
            image = self._get_current_frame()
        except Exception as e:
            self.logger.warning(f"detect_yolo取帧失败: {e}")
            return [], [], [], None
        obj_list, obj_locs, obj_logits, img_with_box, _plot_bgr, cost_ms = self._run_yolo_on_frame(image, object_names)

        if img_with_box is None:
            print("[warn] 未获得推理结果")
            self.last_detection_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            self.last_detection_time = datetime.now()
            self.last_detection_has_object = False
            self.logger.warning(f"detect_yolo无推理结果, target={object_names}")
            return [], [], [], self.last_detection_image

        # 缓存最近一次检测可视化图，供保存函数直接使用
        self.last_detection_image = img_with_box
        self.last_detection_time = datetime.now()
        self.last_detection_has_object = len(obj_list) > 0

        self.logger.info(
            f"detect_yolo target={object_names} matched={len(obj_list)} elapsed_ms={cost_ms:.1f}"
        )
        if obj_list:
            self.logger.info(
                f"detect_detail names={obj_list} locs={obj_locs} confs={[round(v, 3) for v in obj_logits]}"
            )

        return obj_list, obj_locs, obj_logits, img_with_box

    def search_object_detail(self, object_names, mode="quick", yaw_step_deg=40, yaw_hold_s=2.0, camera_hfov_deg=90.0):
        """
        搜索目标并返回结构化信息。
        mode=quick: 先看当前视野，若无目标再旋转搜索，找到首个目标即结束。
        mode=all: 旋转一整圈，统计目标总数和相对朝向角。
        """
        prev_safety_mode = getattr(self, "_safety_motion_mode", "generic")
        self.set_safety_motion_mode("search")
        canonical_name = self._canonical_object_name(object_names)
        mode = (mode or "quick").strip().lower()
        if mode not in ("quick", "all"):
            mode = "quick"

        self.last_search_result_cn = "搜索中"
        start_yaw = float(self.MavList[0].uavAngEular[2])
        all_angles_deg = []

        self.logger.info(
            f"开始搜索目标: query={object_names} canonical={canonical_name} mode={mode} start_yaw={start_yaw:.3f}"
        )

        def collect_angles_from_frame(base_yaw_rad):
            obj_list, obj_locs, obj_logits, img_with_box = self.detect_yolo(canonical_name)
            img_w, _ = img_with_box.size if hasattr(img_with_box, "size") else (640, 480)
            frame_angles = []
            for bbox in obj_locs:
                if not bbox or len(bbox) < 4:
                    continue
                center_x = (bbox[0] + bbox[2]) / 2.0
                pixel_offset = center_x - (img_w / 2.0)
                offset_deg = (pixel_offset / max(img_w / 2.0, 1.0)) * (camera_hfov_deg / 2.0)
                frame_angles.append(offset_deg)
            return frame_angles

        # quick模式先看当前视野
        if mode == "quick":
            init_angles = collect_angles_from_frame(start_yaw)
            if init_angles:
                unique_angles = self._cluster_angles_deg(init_angles, threshold_deg=15.0)
                summary = self._search_summary_line(canonical_name, mode, True, unique_angles)
                self.last_search_result_cn = summary
                self.logger.info(f"search_quick_hit summary={summary}")
                return {
                    "found": True,
                    "mode": mode,
                    "object": canonical_name,
                    "count": len(unique_angles),
                    "angles_deg": unique_angles,
                    "summary": summary,
                }

        # 旋转阶段：quick找到首个目标即结束，all完整旋转一圈
        for yaw_step in range(0, 360, int(yaw_step_deg)):
            new_yaw = start_yaw + math.radians(yaw_step)
            self.MavList[0].SendPosNED(
                self.MavList[0].uavPosNED[0],
                self.MavList[0].uavPosNED[1],
                self.MavList[0].uavPosNED[2],
                new_yaw,
            )
            time.sleep(yaw_hold_s)

            frame_angles = collect_angles_from_frame(new_yaw)
            frame_found = len(frame_angles) > 0
            self.logger.info(f"search_step mode={mode} yaw_step={yaw_step} found={frame_found} hits={len(frame_angles)}")

            if frame_found:
                all_angles_deg.extend(frame_angles)
                if mode == "quick":
                    unique_angles = self._cluster_angles_deg(all_angles_deg, threshold_deg=15.0)
                    summary = self._search_summary_line(canonical_name, mode, True, unique_angles)
                    self.last_search_result_cn = summary
                    self.logger.info(f"search_quick_rotate_hit summary={summary}")
                    return {
                        "found": True,
                        "mode": mode,
                        "object": canonical_name,
                        "count": len(unique_angles),
                        "angles_deg": unique_angles,
                        "summary": summary,
                    }

        if mode == "all" and all_angles_deg:
            unique_angles = self._cluster_angles_deg(all_angles_deg, threshold_deg=15.0)
            summary = self._search_summary_line(canonical_name, mode, True, unique_angles)
            self.last_search_result_cn = summary
            self.logger.info(f"search_all_done summary={summary}")
            return {
                "found": True,
                "mode": mode,
                "object": canonical_name,
                "count": len(unique_angles),
                "angles_deg": unique_angles,
                "summary": summary,
            }

        summary = self._search_summary_line(canonical_name, mode, False, [])
        self.last_search_result_cn = summary
        self.logger.info(f"搜索失败: {summary}")
        return {
            "found": False,
            "mode": mode,
            "object": canonical_name,
            "count": 0,
            "angles_deg": [],
            "summary": summary,
        }
        
        # 不可达分支，仅用于保持 finally 语义一致
        
        

    def search_object(self, object_names, mode="quick"):
        """
        兼容旧接口：返回布尔值（是否找到）。
        新增mode参数用于区分快速搜索与全景搜索。
        """
        prev_safety_mode = getattr(self, "_safety_motion_mode", "generic")
        self.set_safety_motion_mode("search")
        try:
            result = self.search_object_detail(object_names, mode=mode)
            return bool(result.get("found", False))
        finally:
            self.set_safety_motion_mode(prev_safety_mode)

    def cv2_to_base64(self, image, format='.png'):
        # 将OpenCV图像转换为Base64字符串
        success, buffer = cv2.imencode(format, image)
        if not success:
            raise ValueError("图片编码失败，请检查格式参数")
        img_bytes = buffer.tobytes()
        return base64.b64encode(img_bytes).decode('utf-8')

    def look(self):
        # 获取前置摄像头的图像，并通过火山引擎LLM进行图像理解
        self.logger.info("调用look_function进行视觉描述")
        rgb_image = self._get_current_frame()
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
        self.logger.info(f"look_function返回: {str(content)[:120]}")
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
                "yaw_align_timeout": 3.0 if is_cpu else 2.0,  # 对准阶段超时后强制切到推进，避免原地卡死
                "yaw_align_enter_ts": time.monotonic(),
                "yaw_recheck_ts": 0.0,  # 强制切相位后短暂禁止立刻回切

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
        self.logger.info(
            f"approach_loop error=({error_x:.2f},{error_y:.2f}) lp=({s['lp_ex']:.2f},{s['lp_ey']:.2f}) phase={s['phase']}"
        )

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
                    s["yaw_recheck_ts"] = t

                # 防卡死：若长期无法进入对准阈值，强制先推进一段时间，避免任务整体超时
                if (t - s["yaw_align_enter_ts"]) >= s["yaw_align_timeout"]:
                    s["phase"] = "APPROACH"
                    s["yaw_hold_cnt"] = 0
                    s["yaw_recheck_ts"] = t + 1.2
                    self.logger.warning(
                        f"approach_fallback force_phase=APPROACH ex={s['lp_ex']:.2f} ey={s['lp_ey']:.2f}"
                    )

                cmd = (vx, vy, vz, yawrate)

            # ========== 阶段 2：朝向目标推进 ==========
            else:  # "APPROACH"
                # 若偏航又变大，退回对准阶段
                if abs(s["lp_ex"]) > 1.5 * s["yaw_align_tol"] and t >= s["yaw_recheck_ts"]:
                    s["phase"] = "YAW_ALIGN"
                    s["yaw_align_enter_ts"] = t
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
                    self.logger.info(f"approach_cmd cmd={tuple(round(v, 4) for v in smooth_cmd)}")
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

    def face_objective_to_target(self, object_names, max_seconds=15.0, align_tol=25.0, stable_need=3):
        """
        原地转向对准目标，不前进。
        先搜索目标，再根据图像中心偏差调用 faceObjectiveOnly 做原地转向。
        """
        if not object_names:
            print("执行失败：目标名称不能为空")
            self.logger.warning("face_objective_to_target拒绝执行: 目标名称为空")
            return False

        canonical_name = self._canonical_object_name(object_names)

        try:
            if not self.search_object(canonical_name):
                print(f"执行失败：未找到目标 {object_names}")
                self.logger.warning(f"face_objective_to_target搜索失败: {object_names} -> {canonical_name}")
                return False

            start_ts = time.monotonic()
            stable_cnt = 0
            last_abs_error_x = None

            while True:
                if time.monotonic() - start_ts > max_seconds:
                    self.logger.info(
                        f"face_objective_to_target超时退出: target={object_names} elapsed={time.monotonic() - start_ts:.2f}s"
                    )
                    return True

                step_idx = stable_cnt + 1
                obj_list, obj_locs, obj_logits, img_with_box = self.detect_yolo(canonical_name)
                if not obj_list or not obj_locs:
                    print(f"执行失败：未检测到目标 {object_names}")
                    self.logger.warning(f"face_objective_to_target未检测到目标: {object_names} -> {canonical_name}")
                    return False

                bbox = obj_locs[0]
                if len(bbox) < 4:
                    print("执行失败：目标框信息不完整")
                    self.logger.warning(f"face_objective_to_target目标框不完整: {bbox}")
                    return False

                img_w, img_h = img_with_box.size if hasattr(img_with_box, "size") else (640, 480)
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0
                error_x = center_x - img_w / 2.0
                error_y = center_y - img_h / 2.0

                print(f"[FACE] target={object_names} err=({error_x:.1f},{error_y:.1f}) stable={stable_cnt}/{stable_need}")
                self.logger.info(
                    f"face_objective_to_target target={object_names} err=({error_x:.1f},{error_y:.1f}) elapsed={time.monotonic() - start_ts:.2f}s stable={stable_cnt}/{stable_need}"
                )

                if abs(error_x) <= align_tol:
                    stable_cnt += 1
                    self.MavList[0].SendVelFRD(0, 0, 0, 0)
                    self.logger.info(f"face_objective_to_target命中稳定阈值: {object_names} stable={stable_cnt}/{stable_need}")
                    if stable_cnt >= stable_need:
                        self.logger.info(f"face_objective_to_target完成对准: {object_names}")
                        return True
                    time.sleep(0.1)
                    continue

                stable_cnt = 0

                # 若误差在减小但还未到阈值，继续原地朝向；若误差未明显改善，也不要立刻退出，交给时间上限兜底
                if last_abs_error_x is not None and abs(error_x) > last_abs_error_x + 15:
                    self.logger.info(
                        f"face_objective_to_target误差波动: last={last_abs_error_x:.1f}, now={abs(error_x):.1f}"
                    )
                last_abs_error_x = abs(error_x)

                self.faceObjectiveOnly(error_x, error_y)
                time.sleep(0.1)
        except Exception as e:
            print(f"执行失败：原地转向目标异常 {e}")
            self.logger.error(f"face_objective_to_target执行失败: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def approach_objective_to_target(self, object_names, max_seconds=20.0, align_tol=80.0, stable_need=3, box_ratio=1.0/5.0):
        """
        靠近目标：先搜索目标，再循环检测并逼近，直到达到停止条件。
        """
        if not object_names:
            print("执行失败：目标名称不能为空")
            self.logger.warning("approach_objective_to_target拒绝执行: 目标名称为空")
            self.last_search_result_cn = "靠近失败：目标为空"
            return False

        canonical_name = self._canonical_object_name(object_names)

        try:
            if not self.search_object(canonical_name, mode="quick"):
                print(f"执行失败：未找到目标 {object_names}")
                self.logger.warning(f"approach_objective_to_target搜索失败: {object_names} -> {canonical_name}")
                self.last_search_result_cn = f"靠近失败：未找到{canonical_name}"
                return False

            start_ts = time.monotonic()
            stable_cnt = 0

            while True:
                if time.monotonic() - start_ts > max_seconds:
                    self.logger.warning(
                        f"approach_objective_to_target超时: target={canonical_name} elapsed={time.monotonic() - start_ts:.2f}s"
                    )
                    self.last_search_result_cn = f"靠近失败：靠近{canonical_name}超时"
                    self.MavList[0].SendVelFRD(0, 0, 0, 0)
                    return False

                obj_list, obj_locs, obj_logits, img_with_box = self.detect_yolo(canonical_name)
                if not obj_list or not obj_locs:
                    stable_cnt = 0
                    self.logger.warning(f"approach_objective_to_target丢失目标: {canonical_name}")
                    self.MavList[0].SendVelFRD(0, 0, 0, 0)
                    time.sleep(0.08)
                    continue

                bbox = obj_locs[0]
                if len(bbox) < 4:
                    stable_cnt = 0
                    self.MavList[0].SendVelFRD(0, 0, 0, 0)
                    time.sleep(0.08)
                    continue

                img_w, img_h = img_with_box.size if hasattr(img_with_box, "size") else (640, 480)
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0
                error_x = center_x - img_w / 2.0
                error_y = center_y - img_h / 2.0
                box_w = max(0.0, bbox[2] - bbox[0])
                box_h = max(0.0, bbox[3] - bbox[1])
                box_max = max(box_w, box_h)
                need_box = img_w * box_ratio

                stop_now = (box_max >= need_box) and (abs(error_x) <= align_tol) and (abs(error_y) <= align_tol)
                stable_cnt = stable_cnt + 1 if stop_now else 0

                self.logger.info(
                    f"approach_target_loop target={canonical_name} ex={error_x:.1f} ey={error_y:.1f} box={box_max:.1f}/{need_box:.1f} stable={stable_cnt}/{stable_need} stop={stop_now}"
                )

                if stable_cnt >= stable_need:
                    self.MavList[0].SendVelFRD(0, 0, 0, 0)
                    self.last_search_result_cn = f"靠近完成：已逼近{canonical_name}并停稳"
                    self.logger.info(f"approach_done target={canonical_name} result={self.last_search_result_cn}")
                    print(self.last_search_result_cn)
                    return True

                self.approachObjective(error_x, error_y)
                time.sleep(0.08)
        except Exception as e:
            print(f"执行失败：靠近目标异常 {e}")
            self.logger.error(f"approach_objective_to_target执行失败: {e}")
            self.logger.debug(traceback.format_exc())
            self.last_search_result_cn = "靠近失败：执行异常"
            try:
                self.MavList[0].SendVelFRD(0, 0, 0, 0)
            except Exception:
                pass
            return False

    def strike_objective_to_target(
        self,
        object_names,
        max_align_seconds=16.0,
        align_tol=18.0,
        align_tol_y=28.0,
        stable_need=4,
        ram_speed=1.8,
        ram_seconds=0.30,
        extra_forward_m=1.5,
        hit_box_ratio=1.0 / 5.0,
        kp_x=0.00085,
        kd_x=0.00012,
        kp_y=0.0028,
        kd_y=0.00018,
    ):
        """
        打击目标（视觉伺服+速度调度版）：
        先搜索目标，再在前进中持续修正偏航和高度，并按目标大小动态调节前进速度，满足命中条件后穿越并停稳。
        """
        if not object_names:
            print("执行失败：目标名称不能为空")
            self.logger.warning("strike_objective_to_target拒绝执行: 目标名称为空")
            self.last_search_result_cn = "打击失败：目标为空"
            return False

        if ram_speed <= 0 or ram_seconds <= 0 or extra_forward_m < 0 or hit_box_ratio <= 0:
            print("执行失败：打击参数非法")
            self.logger.warning(
                f"strike_objective_to_target参数非法: speed={ram_speed}, ram_seconds={ram_seconds}, extra_m={extra_forward_m}, hit_box_ratio={hit_box_ratio}"
            )
            self.last_search_result_cn = "打击失败：参数非法"
            return False

        canonical_name = self._canonical_object_name(object_names)

        try:
            # 阶段1：先快速搜索目标
            if not self.search_object(canonical_name, mode="quick"):
                print(f"执行失败：未找到目标 {object_names}")
                self.logger.warning(f"strike_objective_to_target搜索失败: {object_names} -> {canonical_name}")
                self.last_search_result_cn = f"打击失败：未找到{canonical_name}"
                return False

            # 阶段2：闭环引导冲刺（边前进边修正偏航）
            guide_start_ts = time.monotonic()
            stable_hit_cnt = 0
            lost_cnt = 0
            step_dt = 0.08
            yaw_max = math.radians(35)
            vz_max = 0.35
            terminal_enter_ratio = 0.45
            prev_error_x = None
            prev_error_y = None
            prev_ts = None
            coarse_align_tol = max(align_tol * 2.5, 40.0)

            def clamp(v, vmin, vmax):
                return vmin if v < vmin else (vmax if v > vmax else v)

            def schedule_forward_speed(box_now, box_need, ex_now, ey_now):
                """
                根据目标框大小与对准程度调度前进速度。
                目标越小越快，越接近命中框越慢；偏离中心时进一步降速。
                """
                if box_need <= 0:
                    return 0.25

                box_ratio = clamp(box_now / box_need, 0.0, 1.5)
                if box_ratio < 0.35:
                    base_v = ram_speed * 1.00
                elif box_ratio < 0.65:
                    base_v = ram_speed * 0.80
                elif box_ratio < 0.90:
                    base_v = ram_speed * 0.55
                else:
                    base_v = ram_speed * 0.28

                align_ratio_x = abs(ex_now) / max(align_tol, 1.0)
                align_ratio_y = abs(ey_now) / max(align_tol_y, 1.0)
                align_penalty = max(0.35, 1.0 - 0.35 * max(align_ratio_x, align_ratio_y))
                return clamp(base_v * align_penalty, 0.20, ram_speed)

            while True:
                elapsed = time.monotonic() - guide_start_ts
                if elapsed > max_align_seconds:
                    self.logger.warning(
                        f"strike_guide超时: target={canonical_name} elapsed={elapsed:.2f}s stable_hit={stable_hit_cnt}/{stable_need}"
                    )
                    self.last_search_result_cn = f"打击失败：引导超时，未命中{canonical_name}"
                    self.MavList[0].SendVelFRD(0, 0, 0, 0)
                    return False

                obj_list, obj_locs, obj_logits, img_with_box = self.detect_yolo(canonical_name)
                if not obj_list or not obj_locs:
                    lost_cnt += 1
                    # 目标短时丢失时先悬停，避免继续直行导致目标更快出画
                    self.MavList[0].SendVelFRD(0.0, 0.0, 0.0, 0.0)
                    if lost_cnt >= 6:
                        self.logger.warning(f"strike_guide丢失目标过久: target={canonical_name}")
                        self.last_search_result_cn = f"打击失败：引导阶段丢失{canonical_name}"
                        self.MavList[0].SendVelFRD(0, 0, 0, 0)
                        return False
                    time.sleep(step_dt)
                    continue

                lost_cnt = 0
                bbox = obj_locs[0]
                if len(bbox) < 4:
                    self.MavList[0].SendVelFRD(0.0, 0.0, 0.0, 0.0)
                    time.sleep(step_dt)
                    continue

                img_w, img_h = img_with_box.size if hasattr(img_with_box, "size") else (640, 480)
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0
                box_w = max(0.0, bbox[2] - bbox[0])
                box_h = max(0.0, bbox[3] - bbox[1])
                box_max = max(box_w, box_h)
                need_box = img_w * hit_box_ratio

                error_x = center_x - img_w / 2.0
                error_y = center_y - img_h / 2.0
                now_ts = time.monotonic()
                dt_vision = now_ts - prev_ts if prev_ts is not None else step_dt
                dt_vision = max(dt_vision, 1e-3)
                d_ex = (error_x - prev_error_x) / dt_vision if prev_error_x is not None else 0.0
                d_ey = (error_y - prev_error_y) / dt_vision if prev_error_y is not None else 0.0
                prev_error_x = error_x
                prev_error_y = error_y
                prev_ts = now_ts

                yawrate = clamp(kp_x * error_x + kd_x * d_ex, -yaw_max, yaw_max)
                vz = clamp(kp_y * error_y + kd_y * d_ey, -vz_max, vz_max)
                well_aligned = (abs(error_x) <= align_tol) and (abs(error_y) <= align_tol_y)
                vx = schedule_forward_speed(box_max, need_box, error_x, error_y)
                # 粗对准阶段先压制前冲，优先把目标拉回画面中心，避免边冲边偏导致长期不过门限。
                if abs(error_x) > coarse_align_tol:
                    yawrate = clamp((kp_x * 1.30) * error_x + kd_x * d_ex, -yaw_max, yaw_max)
                    vx = 0.0
                elif not well_aligned:
                    vx = min(vx, ram_speed * 0.55)
                if abs(error_x) > align_tol * 2.0 or abs(error_y) > align_tol_y * 2.0:
                    vx = min(vx, 0.35)
                self.MavList[0].SendVelFRD(vx, 0.0, vz, yawrate)

                hit_now = well_aligned and (box_max >= need_box)
                stable_hit_cnt = (stable_hit_cnt + 1) if hit_now else 0

                # 目标已经足够大时，直接进入终端短冲，不再强求连续稳定命中
                if box_max >= img_w * terminal_enter_ratio:
                    self.logger.info(
                        f"strike_terminal_enter target={canonical_name} box={box_max:.1f} enter={img_w * terminal_enter_ratio:.1f} stable={stable_hit_cnt}/{stable_need}"
                    )
                    break

                self.logger.info(
                    f"strike_guide target={canonical_name} ex={error_x:.1f}/{align_tol:.1f} ey={error_y:.1f}/{align_tol_y:.1f} box={box_max:.1f}/{need_box:.1f} allow={hit_now} stable={stable_hit_cnt}/{stable_need} vx={vx:.2f} vz={vz:.2f} yawrate={yawrate:.3f}"
                )

                if stable_hit_cnt >= stable_need:
                    break
                time.sleep(step_dt)

            # 阶段3：终端短冲收尾，不再继续依赖检测，避免近距离穿模后失锁
            box_ratio = box_max / max(need_box, 1.0)
            if box_ratio < 1.15:
                base_terminal_seconds = 0.48
            elif box_ratio < 1.50:
                base_terminal_seconds = 0.36
            else:
                base_terminal_seconds = 0.28
            terminal_speed = clamp(ram_speed * 0.60, 0.45, 1.20)

            # 穿越冲刺：在基础短冲之上，叠加一段按额外前进距离换算的时间
            pass_seconds = (extra_forward_m / max(terminal_speed, 0.1)) * 0.65
            terminal_seconds = clamp(base_terminal_seconds + pass_seconds, 0.35, 1.80)

            self.logger.info(
                f"strike_terminal target={canonical_name} box_ratio={box_ratio:.2f} speed={terminal_speed:.2f} base_s={base_terminal_seconds:.2f} pass_s={pass_seconds:.2f} total_s={terminal_seconds:.2f}"
            )
            print(
                f"[STRIKE] target={canonical_name} 进入终端穿越冲刺: speed={terminal_speed:.2f}m/s, duration={terminal_seconds:.2f}s"
            )

            t0 = time.monotonic()
            while True:
                elapsed = time.monotonic() - t0
                if elapsed >= terminal_seconds:
                    break
                self.MavList[0].SendVelFRD(terminal_speed, 0.0, 0.0, 0.0)
                time.sleep(step_dt)

            # 阶段4：刹停
            self.MavList[0].SendVelFRD(0, 0, 0, 0)
            time.sleep(0.15)
            self.MavList[0].SendVelFRD(0, 0, 0, 0)

            self.last_search_result_cn = f"打击完成：已对{canonical_name}执行终端短冲并停稳"
            self.logger.info(f"strike_done target={canonical_name} result={self.last_search_result_cn}")
            print(self.last_search_result_cn)
            return True
        except Exception as e:
            print(f"执行失败：打击目标异常 {e}")
            self.logger.error(f"strike_objective_to_target执行失败: {e}")
            self.logger.debug(traceback.format_exc())
            self.last_search_result_cn = "打击失败：执行异常"
            try:
                self.MavList[0].SendVelFRD(0, 0, 0, 0)
            except Exception:
                pass
            return False

    def save_detection_image(self, output_dir=None, file_name=None, use_latest=False):
        """
        保存带有检测结果的摄像头图片。
        默认实时触发一次检测并保存；
        当use_latest=True时优先保存最近一次检测缓存图（若缓存为空则自动触发一次检测）。
        :return: 保存成功返回文件路径，失败返回None。
        """
        img_with_box = None

        has_object = False

        if use_latest and self.last_detection_image is not None:
            img_with_box = self.last_detection_image
            has_object = bool(self.last_detection_has_object)
            self.logger.info("save_detection_image使用缓存检测图")
        else:
            obj_list, _, _, img_with_box = self.detect_yolo("")
            has_object = len(obj_list) > 0
            self.logger.info("save_detection_image实时触发检测")

        if img_with_box is None:
            # 无检测结果时仍保存当前原图（需求2.A）
            img_with_box = self._get_current_frame()
            has_object = False

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "saved_detections")
        os.makedirs(output_dir, exist_ok=True)

        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"detection_{timestamp}.png"

        file_path = os.path.join(output_dir, file_name)

        try:
            if isinstance(img_with_box, np.ndarray):
                ok = cv2.imwrite(file_path, img_with_box)
                if not ok:
                    print(f"保存失败：cv2.imwrite返回False，路径={file_path}")
                    self.logger.error(f"保存失败: cv2.imwrite返回False, path={file_path}")
                    return None
            elif isinstance(img_with_box, Image.Image):
                img_with_box.save(file_path)
            else:
                print(f"保存失败：不支持的图像类型 {type(img_with_box)}")
                self.logger.error(f"保存失败: 不支持图像类型 {type(img_with_box)}")
                return None
        except Exception as e:
            print(f"保存失败：{e}")
            self.logger.error(f"保存失败: {e}")
            return None

        if has_object:
            print(f"检测结果图片已保存：{file_path}")
            self.logger.info(f"保存成功(含目标): {file_path}")
        else:
            print(f"未检测到目标，已保存当前摄像头图片：{file_path}")
            self.logger.info(f"保存成功(无目标): {file_path}")
        return file_path
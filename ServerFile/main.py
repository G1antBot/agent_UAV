# 主函数：利用Smolagents与火山引擎LLM实现UAV_Agent
import time
import math
import sys
sys.path.append(r"D:\Rflysim\RflySimAPIs\RflySimSDK\vision")
from OpenAI_api_Mavlink_Agent import OpenAI_APIs
from Communication_Mavlink import BodyCommMavlink
from runtime_logger import init_runtime_logger, get_runtime_logger


def build_mock_mocap_pose_provider():
    """构造一个最小可跑的动捕回调，用于未接入真实动捕时联调桥接链路。"""
    t0 = time.monotonic()

    def _provider():
        t = time.monotonic() - t0
        x = 0.0
        y = 0.0
        z = 1.0
        roll = 0.0
        pitch = 0.0
        yaw = 0.2 * math.sin(0.3 * t)
        return (x, y, z, roll, pitch, yaw)

    return _provider

if __name__ == '__main__':
    init_runtime_logger()
    logger = get_runtime_logger("main")
    logger.info("主程序启动")
    Comm_api = None
    try:
        # 1) 初始化通信
        Comm_api = BodyCommMavlink()
        logger.info("通信模块初始化完成")
        time.sleep(2)

        # real_mocap模式下默认不注入mock；仅在配置显式允许时注入。
        if getattr(Comm_api, "run_mode", "sim") == "real_mocap":
            if bool(getattr(Comm_api, "is_mock_mocap_allowed", lambda: False)()):
                Comm_api.set_mocap_pose_provider(build_mock_mocap_pose_provider())
                logger.warning("real_mocap模式: allow_mock_mocap_for_debug=true，已注入mock位姿回调用于联调")
            else:
                logger.warning("real_mocap模式: 默认禁用mock位姿注入，请接入真实动捕回调或在Config中显式开启allow_mock_mocap_for_debug")

        # 2) 预检（链路/状态）
        preflight = Comm_api.preflight_check()
        logger.info(f"预检结果: {preflight}")
        if not bool(preflight.get("ok", False)):
            raise RuntimeError("启动预检失败，请先修复链路/状态问题")

        # 3) 启动位姿桥接（实飞模式）
        if getattr(Comm_api, "run_mode", "sim") == "real_mocap":
            bridge_ok = Comm_api.start_mocap_bridge(hz=30.0)
            if not bridge_ok:
                raise RuntimeError("real_mocap模式下动捕桥接启动失败")
            logger.info("real_mocap模式: 动捕桥接已启动")

            preview_cfg = getattr(Comm_api, "_preview_cfg", {}) if hasattr(Comm_api, "_preview_cfg") else {}
            if bool(preview_cfg.get("auto_start", True)):
                preview_ok = Comm_api.start_realtime_preview()
                if not preview_ok:
                    logger.warning("real_mocap模式: 实时预览启动失败，将继续执行飞控任务")
                else:
                    logger.info("real_mocap模式: 实时预览已启动")

        # 4) 进入交互任务循环
        MavList, VehilceNum, Error2UE4Map = Comm_api.GetBodyMavList()
        logger.info(f"无人机信息: VehilceNum={VehilceNum}, Error2UE4Map_len={len(Error2UE4Map)}")
        logger.info(f"安全摘要: {Comm_api.get_safety_summary()}")

        chat_api = OpenAI_APIs(
            MavList,
            VehilceNum,
            Comm_api.detect_yolo,
            Comm_api.approachObjective,
            Comm_api.look,
            Comm_api.search_object,
            Comm_api.save_detection_image,
            Comm_api.face_objective_to_target,
            Comm_api.strike_objective_to_target
        )
        logger.info("进入主控制循环")
        chat_api.Main_Control()
    finally:
        if Comm_api is not None:
            try:
                Comm_api.stop_mocap_bridge()
            except Exception:
                pass
            try:
                Comm_api.close_image_source()
            except Exception:
                pass
        logger.info("主程序退出")

    '''
    # 注释部分：用于测试detect_yolo方法的性能
    print("start thread_comm")
    while True:
        start_time = time.time()
        # 调用detect_yolo方法，检测目标为"airplane"
        Comm_api.detect_yolo("airplane")
        # 打印每次调用的时间
        print(time.time() - start_time)
        # time.sleep(5)
    '''

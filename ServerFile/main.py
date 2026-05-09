# 主函数：利用Smolagents与火山引擎LLM实现UAV_Agent
import time
import math
import sys
sys.path.append(r"D:\Rflysim\RflySimAPIs\RflySimSDK\vision")
from OpenAI_api_Mavlink_Agent import OpenAI_APIs
from Communication_Mavlink import  BodyCommMavlink
from runtime_logger import init_runtime_logger, get_runtime_logger
from MocapClient import OptiTrackClient


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


        # 2) 预检（链路/状态）
        preflight = Comm_api.preflight_check()
        logger.info(f"预检结果: {preflight}")
        if not bool(preflight.get("ok", False)):
            raise RuntimeError("启动预检失败，请先修复链路/状态问题")

        # 3) 启动位姿桥接与实时预览（实飞模式）
        run_mode = getattr(Comm_api, "run_mode", "sim")
        opti_client = None
        if run_mode == "real_mocap":
            mocap_cfg = getattr(Comm_api, "_runtime_cfg", {}).get("mocap", {}) if getattr(Comm_api, "_runtime_cfg", {}) else {}
            mocap_ip = mocap_cfg.get("multicast_ip", "239.255.42.99")
            mocap_port = mocap_cfg.get("port", 1511)
            rb_id = mocap_cfg.get("rigid_body_id", 1)

            opti_client = OptiTrackClient(multicast_ip=mocap_ip, port=mocap_port, rigid_body_id=rb_id)
            opti_client.start()
            
            # 注入纯Python动捕回调
            Comm_api.set_mocap_pose_provider(opti_client.get_latest_pose)

            bridge_ok = Comm_api.start_mocap_bridge(hz=30.0)
            if not bridge_ok:
                raise RuntimeError("real_mocap模式下动捕桥接启动失败")
            logger.info("real_mocap模式: 动捕桥接已启动")

        if run_mode in ("real_mocap", "real_optical"):
            preview_cfg = getattr(Comm_api, "_preview_cfg", {}) if hasattr(Comm_api, "_preview_cfg") else {}
            if bool(preview_cfg.get("auto_start", True)):
                preview_ok = Comm_api.start_realtime_preview()
                if not preview_ok:
                    logger.warning(f"{run_mode}模式: 实时预览启动失败，将继续执行飞控任务")
                else:
                    logger.info(f"{run_mode}模式: 实时预览已启动")
        elif run_mode == "sim":
            sim_preview_cfg = getattr(Comm_api, "_sim_preview_cfg", {}) if hasattr(Comm_api, "_sim_preview_cfg") else {}
            if bool(sim_preview_cfg.get("auto_start", True)):
                preview_ok = Comm_api.start_sim_preview()
                if not preview_ok:
                    logger.warning("sim模式: 实时预览启动失败，将继续执行飞控任务")
                else:
                    logger.info("sim模式: 实时预览已启动")

        # 4) 进入交互任务循环
        MavList, VehilceNum, Error2UE4Map = Comm_api.GetBodyMavList()
        if len(MavList) > 0:
            MavList[0].move_with_speed = Comm_api.move_with_speed

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
        
        # 建立底层中断桥梁
        if hasattr(Comm_api, 'set_interrupt_check'):
            Comm_api.set_interrupt_check(lambda: getattr(chat_api, "is_interrupted", False))

        # 围栏连续触发超限时自动中断当前任务（与 test_exp2 保持一致）
        Comm_api._interrupt_set_callback = lambda: setattr(chat_api, "is_interrupted", True)

        logger.info("进入主控制循环")
        chat_api.Main_Control()
    finally:
        if 'opti_client' in locals() and opti_client is not None:
            try:
                opti_client.stop()
            except Exception:
                pass
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

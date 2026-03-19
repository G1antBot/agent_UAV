# 主函数：利用Smolagents与火山引擎LLM实现UAV_Agent
import time
from OpenAI_api_Mavlink_Agent import OpenAI_APIs
from Communication_Mavlink import BodyCommMavlink

if __name__ == '__main__':
    # 创建BodyCommMavlink类的实例，初始化无人机的Mavlink通信
    Comm_api = BodyCommMavlink()
    # 暂停5秒，确保通信初始化完成
    time.sleep(5)
    # 获取无人机列表、无人机数量和坐标偏移量
    MavList, VehilceNum, Error2UE4Map = Comm_api.GetBodyMavList()
    # 创建OpenAI_APIs类的实例，传入无人机列表、数量和功能函数
    chat_api = OpenAI_APIs(
        MavList,
        VehilceNum,
        Comm_api.detect_dino,  # 目标检测函数
        Comm_api.approachObjective,  # 接近目标函数
        Comm_api.look,  # 前置摄像头图像处理函数
        Comm_api.search_object  # 搜索目标函数
    )
    # 启动主控制逻辑，用户可以通过自然语言输入指令，模型生成相应的控制代码并执行
    chat_api.Main_Control()

    '''
    # 注释部分：用于测试detect_dino方法的性能
    print("start thread_comm")
    while True:
        start_time = time.time()
        # 调用detect_dino方法，检测目标为"airplane"
        Comm_api.detect_dino("airplane")
        # 打印每次调用的时间
        print(time.time() - start_time)
        # time.sleep(5)
    '''

import time
from Communication_Mavlink import BodyCommMavlink

if __name__ == '__main__':
    print("启动调试模式...")

    # 初始化通信系统
    Comm_api = BodyCommMavlink()
    time.sleep(5)  # 等待摄像头初始化

    print("开始调试图像处理...")
    Comm_api.debug_image_processing()

    print("调试完成，检查生成的图片文件:")
    print("- debug_ue4_raw.jpg")
    print("- debug_ue4_processed.jpg")
    print("- temp_test_image.jpg")
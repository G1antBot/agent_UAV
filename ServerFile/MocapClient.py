import threading
import time
import math
import optirx as rx
import logging

def quat_to_euler(qx, qy, qz, qw):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    """
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Use standard asin to avoid complex math sqrt issues
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class OptiTrackClient:
    def __init__(self, multicast_ip="239.255.42.99", port=1511, rigid_body_id=1):
        self.multicast_ip = multicast_ip
        self.port = port
        self.rigid_body_id = rigid_body_id
        
        self.logger = logging.getLogger("uav_agent.mocap")
        
        # Latest pose cache: (x, y, z, roll, pitch, yaw)
        self._latest_pose = None
        self._pose_lock = threading.Lock()
        
        self._running = False
        self._thread = None
        
        self.logger.info(f"OptiTrackClient 初始化: multicast={self.multicast_ip}:{self.port}, target_id={self.rigid_body_id}")

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, name="OptiTrackRecv", daemon=True)
        self._thread.start()
        self.logger.info("OptiTrack 接收线程已启动")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.logger.info("OptiTrack 接收线程已停止")

    def _recv_loop(self):
        # Create socket
        try:
            ds = rx.mkdatasock(ip_address="0.0.0.0", multicast_address=self.multicast_ip, port=self.port)
            self.logger.info("成功连接到 OptiTrack Multicast 网络")
        except Exception as e:
            self.logger.error(f"无法建立 OptiTrack UDP socket: {e}")
            self._running = False
            return

        while self._running:
            try:
                data = ds.recv(32768)
                packet = rx.unpack(data)
                
                # Check for rigid body data
                if type(packet) is rx.FrameOfData:
                    rb_dict = packet.rigid_bodies
                    if self.rigid_body_id in rb_dict:
                        rb = rb_dict[self.rigid_body_id]
                        # OptiTrack returns position in meters
                        x, y, z = rb.position
                        qx, qy, qz, qw = rb.orientation
                        
                        roll, pitch, yaw = quat_to_euler(qx, qy, qz, qw)
                        
                        with self._pose_lock:
                            self._latest_pose = (x, y, z, roll, pitch, yaw)
            except Exception as e:
                self.logger.error(f"OptiTrack 数据解析异常: {e}")
                time.sleep(0.5)

    def get_latest_pose(self):
        """
        供 Communication_Mavlink 调用，返回最新位姿。
        若超过一定时间未收到数据或未初始化，可以根据需求返回 None 或旧数据。
        目前返回最近一次收到的 (x, y, z, roll, pitch, yaw) 或 None。
        """
        with self._pose_lock:
            return self._latest_pose

import socket
import time

class WSG50:

    def __init__(self, host='192.168.1.20', port=1000, timeout=2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.connect()

    def connect(self):
        """建立 TCP 连接"""
        if self.sock:
            self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        try:
            self.sock.connect((self.host, self.port))
            self._flush()
            print(f"[WSG50] Connected to {self.host}:{self.port}")
        except Exception as e:
            raise ConnectionError(f"WSG50 连接失败: {e}")

    def _flush(self):
        """清空缓冲区"""
        try:
            while True:
                data = self.sock.recv(1024)
                if not data:
                    break
        except socket.timeout:
            pass

    def _send(self, cmd):
        """发送 ASCII 命令（阻塞）"""
        if not self.sock:
            raise ConnectionError("WSG50 未连接")
        self.sock.sendall((cmd + '\n').encode('ascii'))
        # 阻塞等待动作完成，夹爪动作通常自动完成，无需接收返回
        time.sleep(0.01)  # 短延时防止发送过快

    def close(self):
        """关闭连接"""
        if self.sock:
            self.sock.close()
            self.sock = None
            print("[WSG50] Connection closed")

    # ---------------- 基本动作 ----------------

    def move(self, width, speed=50, force=50):
        """
        阻塞式移动夹爪到指定宽度（mm）
        width: 0-110
        speed: 0-100
        force: 0-100
        """
        width = max(0, min(110, width))
        speed = max(0, min(100, speed))
        force = max(0, min(100, force))
        self._send(f"move {width} {speed} {force}")
        # 阻塞等待动作完成（夹爪硬件动作时间约 0.5~1 秒，可根据需求调整）
        time.sleep(0.8)

    def grip(self):
        """闭合夹爪抓取"""
        self.move(0)

    def release(self):
        """打开夹爪释放"""
        self.move(110)

# ---------------- 使用示例 ----------------
if __name__ == "__main__":
    try:
        gripper = WSG50("192.168.1.20")

        print("打开夹爪...")
        gripper.release()
        print("夹爪已打开")

        time.sleep(1)

        print("闭合夹爪...")
        gripper.grip()
        print("夹爪已闭合")

    except Exception as e:
        print("出错：", e)

    finally:
        gripper.close()
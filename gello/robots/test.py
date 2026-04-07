import socket
import time

GRIPPER_IP = "192.168.1.20"
PORT = 1000

def send_command(sock, cmd):
    print(f">>> Sending: {cmd.strip()}")
    sock.sendall(cmd.encode())
    time.sleep(0.1)  # 给点时间响应
    try:
        response = sock.recv(1024).decode().strip()
        print(f"<<< Received: {response}")
    except:
        print("No response or timeout")
    print("-" * 40)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(5.0)
    try:
        s.connect((GRIPPER_IP, PORT))
        print(f"Connected to {GRIPPER_IP}:{PORT}")
        
        # 步骤1: 发送 ACK 清醒它（可选，但有用）
        send_command(s, "ACK\n")
        
        # 步骤2: 先 homing（必须先做一次！否则 MOVE 不响应）
        # 如果已经 homing 过，可以注释掉这行，节省时间
        send_command(s, "HOME\n")   # 会动几秒，观察夹爪是否回零/全开
        time.sleep(5)  # 等 homing 完成
        
        # 步骤3: 测试 MOVE 全开 (≈109mm)
        send_command(s, "MOVE(109, 400)\n")   # 位置 mm, 速度 mm/s
        time.sleep(3)
        
        # 步骤4: 测试 MOVE 闭合一点 (≈50mm)
        send_command(s, "MOVE(50, 400)\n")
        time.sleep(3)
        
        # 步骤5: 测试 MOVE 几乎全闭 (软限位防撞，0.8mm)
        send_command(s, "MOVE(5, 200)\n")     # 慢速测试
        
        print("测试完成！如果夹爪动了，说明硬件+网络 OK。")
        
    except Exception as e:
        print(f"连接/命令失败: {e}")
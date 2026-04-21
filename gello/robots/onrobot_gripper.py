import time
import onRobot.gripper as gripper   # ← 推荐这种导入方式

ROBOT_IP = "192.168.1.101"
GRIPPER_ID = 0   # 大多数情况是 0，确认方法：在 PolyScope 的 Log 标签里运行 OnRobot 组件看 ID

def test_rg2():
    print("使用 RyanPaulMcKenna/onRobot 库控制 OnRobot RG2 夹爪")

    # 创建夹爪对象（推荐写法）
    rg = gripper.RG2(GRIPPER_ID)     # 或者试试 gripper.RG2(GRIPPER_ID) 如果上面不行

    try:
        while True:
            print("\n→ 全开夹爪 (宽度 ≈110mm, 力 40N)")
            rg.rg_grip(110.0, 40.0)
            time.sleep(3)

            print("→ 用力闭合 (宽度 0mm, 力 40N)")
            rg.rg_grip(0.0, 40.0)
            time.sleep(3)

            print("→ 指定宽度抓取 (宽度 50mm, 力 30N)")
            rg.rg_grip(50.0, 30.0)
            time.sleep(3)

            # 读取当前宽度（验证是否通信正常）
            try:
                width = rg.get_rg_width()
                print(f"当前夹爪宽度: {width:.2f} mm")
            except Exception as e:
                print(f"读取宽度失败: {e} （夹爪如果在动，可暂时忽略）")

    except KeyboardInterrupt:
        print("\n测试被手动中断 (Ctrl+C)")
    finally:
        print("安全释放：全开夹爪...")
        rg.rg_grip(110.0, 40.0)
        time.sleep(1)
        print("测试结束")

if __name__ == "__main__":
    test_rg2()
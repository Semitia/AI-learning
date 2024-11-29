import pybullet as p
import pybullet_data
import time

# 连接到服务器（GUI)
physics_client = p.connect(p.GUI)
# 无GUI版本
# physics_client = p.connect(p.DIRECT)

# 添加 PyBullet 自带的搜索路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# 设置仿真参数
p.setGravity(0, 0, -9.8)  # 设置重力
# 加载地面和KUKA
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("panda/model.urdf", [0, 0, 0])
# 仿真
for _ in range(240):
    p.stepSimulation()
    time.sleep(1 / 240)  # 控制仿真速度
# 断开
p.disconnect()

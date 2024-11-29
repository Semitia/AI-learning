import pybullet as p
import time
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# 加载 URDF
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("simple_robot.urdf", basePosition=[0, 0, 0.5],
                      baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))

# 查看机器人信息
num_joints = p.getNumJoints(robot_id)
print(f"机器人有 {num_joints} 个关节")
# 打印每个关节的信息
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    print(f"关节[{i}] - 名称: {joint_info[1].decode('utf-8')}, 类型: {joint_info[2]}")

p.setGravity(0, 0, -9.8)
try:
    while True:
        p.stepSimulation()
        time.sleep(1/240)
except KeyboardInterrupt:
    p.disconnect()

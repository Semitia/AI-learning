import pybullet as p
import pybullet_data
import time

# 初始化
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载 URDF
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("robot.urdf", basePosition=[2, 3, 0.2],
                      baseOrientation=p.getQuaternionFromEuler([0, 0, 0.5]))

# 设置重力
p.setGravity(0, 0, -9.8)

# 初始化一些变换参数
base_positions = [[2, 3, 0.2], [2, 3, 0.5], [2, 3, 1.0], [1, 1, 1]]  # 位置序列
base_orientations = [
    p.getQuaternionFromEuler([0, 0, 0.5]),
    p.getQuaternionFromEuler([0, 0, 1.0]),
    p.getQuaternionFromEuler([0, 0, 1.57]),
    p.getQuaternionFromEuler([0, 0, 2.0])
]  # 方向序列

# 动态调整关节角度
joint_index = 1
joint_angles = [0, 0.5, 1.0, 1.57]

try:
    step = 0
    while True:
        # 每隔一定步数执行一次变换
        if step % 240 == 0:
            # 更新关节角度
            new_joint_angle = joint_angles[step // 240 % len(joint_angles)]
            p.resetJointState(robot_id, joint_index, targetValue=new_joint_angle)
            print(f"设置关节 {joint_index} 角度: {new_joint_angle}")

            # 获取并打印当前状态
            base_position, base_orientation = p.getBasePositionAndOrientation(robot_id)
            print(f"当前根链接位置: {base_position}")
            print(f"当前根链接方向: {base_orientation}")
            joint_state = p.getJointState(robot_id, joint_index)
            print(f"关节 {joint_index} 的角度: {joint_state[0]}")

        p.stepSimulation()
        time.sleep(1 / 240)
        step += 1
except KeyboardInterrupt:
    p.disconnect()

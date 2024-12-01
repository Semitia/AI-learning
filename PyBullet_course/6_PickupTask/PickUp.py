import pybullet as p
import pybullet_data
import time
import math
import numpy as np


# 初始化 PyBullet
physics_client = p.connect(p.GUI)
urdfRoot = pybullet_data.getDataPath()
p.setAdditionalSearchPath(urdfRoot)
p.setGravity(0, 0, -10)
time_step = 1/240
p.setTimeStep(time_step)
p.resetDebugVisualizerCamera(1.5, 180, -41, [-0.52, -0.2, -0.33])
# 加载物品
plane_id = p.loadURDF("plane.urdf", [0, 0, -1])
table_id = p.loadURDF("table/table.urdf", -0.5000000, 0.00000, -.820000, 0.000000, 0.000000, 0.0, 1.0)
tray_id = p.loadURDF("tray/tray.urdf", -0.60000, 0.075000, -0.190000, 0.000000, 0.000000, 1.000000, 0.000000)
ang = 3.14 * 0.5
orn = p.getQuaternionFromEuler([0, 0, ang])
block_id = p.loadURDF("block.urdf", -0.6, 0, -0.15, orn[0], orn[1], orn[2], orn[3])
kuka_id = p.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")
kuka_id = kuka_id[0]
num_joints = p.getNumJoints(kuka_id)
print("num_joints: ", num_joints)
for i in range(num_joints):
    print(p.getJointInfo(kuka_id, i))
# 参数设置
finger_Force = 2
end_effector_index = 7
jd = [
    0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
    0.00001, 0.00001, 0.00001, 0.00001
]


# 定义机械臂运动函数
def set_pose(target_pos, target_ori, max_iters=120, threshold=0.01):
    target_ori = p.getQuaternionFromEuler(target_ori)
    for _ in range(max_iters):
        # 计算逆运动学以得到关节角
        joint_positions = p.calculateInverseKinematics(kuka_id, end_effector_index, target_pos, target_ori)
        # print("len(joint_positions): ", len(joint_positions))
        for j in range(end_effector_index):
            p.setJointMotorControl2(kuka_id, j, p.POSITION_CONTROL, targetPosition=joint_positions[j])
        p.stepSimulation()
        time.sleep(1 / 240)

        # 获取当前末端位置
        link_state = p.getLinkState(kuka_id, end_effector_index)
        current_position = link_state[0]
        current_orientation = p.getEulerFromQuaternion(link_state[1])
        current_pose = np.array(list(current_position) + list(current_orientation))
        target_pose_array = np.array(list(target_pos) + list(p.getEulerFromQuaternion(target_ori)))
        distance = np.linalg.norm(current_pose - target_pose_array)
        # print("current_position: ", current_position)
        # print("current_orientation: ", current_orientation)
        # print("distance: ", distance)
        # 检查是否接近目标位置
        if distance < threshold:
            break


def set_gripper(gripper_angle):
    # p.setJointMotorControl2(kuka_id,
    #                         7,
    #                         p.POSITION_CONTROL,
    #                         targetPosition=gripper_angle,
    #                         force=self.maxForce)
    p.setJointMotorControl2(kuka_id,
                            8,
                            p.POSITION_CONTROL,
                            targetPosition=-gripper_angle,
                            force=finger_Force)
    p.setJointMotorControl2(kuka_id,
                            11,
                            p.POSITION_CONTROL,
                            targetPosition=gripper_angle,
                            force=finger_Force)
    for _ in range(50):
        p.stepSimulation()
        time.sleep(1 / 240)


set_pose([-0.6, 0, 0.1], [0, math.pi, 0])
set_gripper(0.3)
set_pose([-0.6, 0, 0.055], [0, math.pi, 0])
set_gripper(0.0)
if p.getContactPoints(block_id, kuka_id):
    print("Block is grasped!")
    set_pose([-0.6, 0, 0.2], [0, math.pi, 0])
else:
    print("Not find block!")

while True:
    p.stepSimulation()
    time.sleep(1 / 240)

p.disconnect()

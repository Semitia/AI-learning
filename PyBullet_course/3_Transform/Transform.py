import pybullet as p
import pybullet_data
import time

# 初始化
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载 URDF
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("robot.urdf", basePosition=[2, 3, 0.2],
                      baseOrientation=p.getQuaternionFromEuler([0, 0, 0.5]))
# 查看机器人信息
num_joints = p.getNumJoints(robot_id)
num_links = num_joints + 1
print(f"机器人有 {num_joints} 个关节")
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    print(f"关节[{i}] - 名称: {joint_info[1].decode('utf-8')}, 类型: {joint_info[2]}")

# 获取根链接的位置和方向
base_position, base_orientation = p.getBasePositionAndOrientation(robot_id)
print(f"位置 (x, y, z): {base_position}")
print(f"方向 (四元数): {base_orientation}")

# 获取特定链接的位置和方向
link_index = 1
link_state = p.getLinkState(robot_id, link_index)
link_position_in_parent_frame = link_state[0]  # 链接的局部位置
link_orientation_in_parent_frame = link_state[1]  # 链接的局部方向 (四元数)
linear_velocity_in_parent_frame = link_state[2]
angular_velocity_in_parent_frame = link_state[3]
link_position = link_state[4]  # 链接的全局位置
link_orientation = link_state[5]  # 链接的全局方向 (四元数)
print(f"链接 {link_index} 在父坐标系中的位置: {link_position_in_parent_frame}")
print(f"链接 {link_index} 在父坐标系中的方向: {link_orientation_in_parent_frame}")
print(f"链接 {link_index} 在父坐标系中的线速度: {linear_velocity_in_parent_frame}")
print(f"链接 {link_index} 在父坐标系中的角速度: {angular_velocity_in_parent_frame}")
print(f"链接 {link_index} 在全局坐标系中的位置: {link_position}")
print(f"链接 {link_index} 在全局坐标系中的方向: {link_orientation}")

# 将根链接移动到指定位置并设置方向
new_position = [1, 1, 1]
new_orientation = p.getQuaternionFromEuler([0, 0, 1.57])  # 将欧拉角转为四元数
p.resetBasePositionAndOrientation(robot_id, new_position, new_orientation)
# 验证设置是否生效
base_position, base_orientation = p.getBasePositionAndOrientation(robot_id)
print(f"新的位置: {base_position}")
print(f"新的方向: {base_orientation}")

joint_index = 1  # 假设控制关节 2
target_angle = 1.57  # 弧度
p.resetJointState(robot_id, joint_index, targetValue=target_angle)

# 验证设置是否生效
joint_state = p.getJointState(robot_id, joint_index)
print(f"关节 {joint_index} 的角度: {joint_state[0]}")

p.disconnect()

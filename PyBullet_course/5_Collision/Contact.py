import pybullet as p
import pybullet_data
import time


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 设置资源路径
p.loadURDF("plane.urdf")  # 加载平面
robot_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0.5])  # 加载机器人
obstacle_id = p.loadURDF("cube_small.urdf", basePosition=[0.5, 0, 0.5])  # 加载障碍物
p.setGravity(0, 0, -9.8)

# 碰撞形状信息
print("\n==== 碰撞形状信息 (getCollisionShapeData) ====")
collision_shape_robot = p.getCollisionShapeData(robot_id, linkIndex=-1)  # 获取机器人基座的碰撞形状
collision_shape_obstacle = p.getCollisionShapeData(obstacle_id, linkIndex=-1)  # 获取障碍物的碰撞形状

# 机器人碰撞形状
print("机器人碰撞形状信息:")
for shape in collision_shape_robot:
    print(f"  碰撞形状类型: {shape[2]}")  # 类型：如 box, sphere, cylinder
    print(f"  碰撞形状尺寸: {shape[3]}")  # 尺寸

# 障碍物碰撞形状
print("障碍物碰撞形状信息:")
for shape in collision_shape_obstacle:
    print(f"  碰撞形状类型: {shape[2]}")
    print(f"  碰撞形状尺寸: {shape[3]}")

cnt = 0
try:
    while True:
        time.sleep(1 / 240)
        p.stepSimulation()
        if cnt % 120 == 0:
            # 两物体之间接触点信息
            print("==== 接触点信息 (getContactPoints) ====")
            contact_points = p.getContactPoints(robot_id, obstacle_id)
            if len(contact_points) == 0:
                print("机器人和障碍物之间没有接触点。")
            else:
                for i, point in enumerate(contact_points):
                    print(f"接触点 {i + 1}:")
                    print(f"  位置: {point[5]}")
                    print(f"  法向量: {point[7]}")
                    print(f"  接触深度: {point[8]}")

            # 两物体之间最近点信息
            print("\n==== 最近点信息 (getClosestPoints) ====")
            closest_points = p.getClosestPoints(robot_id, obstacle_id, distance=1.0)  # 设置查询范围
            if len(closest_points) == 0:
                print("机器人和障碍物之间的距离大于 1.0 米。")
            else:
                # 最近的点（getClosestPoints 的返回值已经按距离排序，第一个点即最近点）
                closest_point = closest_points[0]
                print("最近点信息:")
                print(f"  机器人最近点位置: {closest_point[5]}")
                print(f"  障碍物最近点位置: {closest_point[6]}")
                print(f"  距离: {closest_point[8]}")
        cnt = cnt + 1
except KeyboardInterrupt:
    p.disconnect()

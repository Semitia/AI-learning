import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0])
p.setGravity(0, 0, -9.8)
time.sleep(1)

joint_index = 0
# 位置控制
print("位置控制")
p.setJointMotorControl2(
    bodyUniqueId=robot_id,
    jointIndex=joint_index,
    controlMode=p.POSITION_CONTROL,
    targetPosition=1.0,
    force=20
)
for _ in range(240):
    p.stepSimulation()
    time.sleep(1/240)

# 速度控制
print("速度控制")
p.setJointMotorControl2(
    bodyUniqueId=robot_id,
    jointIndex=joint_index,
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity=2.0,
    force=20
)
for _ in range(240):
    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()

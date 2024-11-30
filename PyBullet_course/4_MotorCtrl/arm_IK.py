import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", [0, 0, -0.3])
kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
if numJoints != 7:
    exit()

# 零空间下限位
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
# 零空间上限位
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
# 关节范围
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# 关节初始位置
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
# 关节阻尼
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for i in range(numJoints):
    p.resetJointState(kukaId, i, rp[i])
p.setGravity(0, 0, 0)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 1
useOrientation = 1
useSimulation = 1  # 运动仿真
useRealTimeSimulation = 0  # 实时仿真
ikSolver = 0
p.setRealTimeSimulation(useRealTimeSimulation)
trailDuration = 8  # 轨迹持续时间
pos = [0, 0, 0]
orn = [0, 0, 0, 1]
i = 0
while True:
    i = i + 1
    if useRealTimeSimulation:
        dt = datetime.now()
        t = (dt.second / 60.) * 2. * math.pi
    else:
        t = t + 0.01

    if useSimulation and useRealTimeSimulation == 0:
        p.stepSimulation()
        time.sleep(1/240)

    for i in range(1):
        pos = [-0.4, 0.2 * math.cos(t), 0. + 0.2 * math.sin(t)]
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])

    if useNullSpace == 1:
        if useOrientation == 1:
            jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos, orn, ll, ul, jr, rp)
        else:
            jointPoses = p.calculateInverseKinematics(kukaId,
                                                      kukaEndEffectorIndex,
                                                      pos,
                                                      lowerLimits=ll,
                                                      upperLimits=ul,
                                                      jointRanges=jr,
                                                      restPoses=rp)
    else:
        if useOrientation == 1:
            jointPoses = p.calculateInverseKinematics(kukaId,
                                                      kukaEndEffectorIndex,
                                                      pos,
                                                      orn,
                                                      jointDamping=jd,
                                                      solver=ikSolver,
                                                      maxNumIterations=100,   # 最大迭代次数
                                                      residualThreshold=.01)  # 残差阈值
        else:
            jointPoses = p.calculateInverseKinematics(kukaId,
                                                      kukaEndEffectorIndex,
                                                      pos,
                                                      solver=ikSolver)      # 求解器

    if useSimulation:
        for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)
    else:
        for i in range(numJoints):
            p.resetJointState(kukaId, i, jointPoses[i])

    ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
    if hasPrevPose:
        p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)    # 目标位置轨迹线
        p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)   # 实际位置轨迹线
    prevPose = pos
    prevPose1 = ls[4]
    hasPrevPose = 1

p.disconnect()

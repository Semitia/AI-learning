import os
import math
import pybullet as p
import pybullet_data


class Kuka:
    def __init__(self, urdf_root_path=pybullet_data.getDataPath(), time_step=0.01):
        self.urdf_root_path = urdf_root_path
        self.time_step = time_step
        self.max_velocity = 0.35
        self.max_force = 200.0
        self.finger_a_force = 2.0
        self.finger_b_force = 2.5
        self.finger_tip_force = 2.0
        self.use_inverse_kinematics = True
        self.use_simulation = True
        self.use_null_space = True
        self.use_orientation = True
        self.kuka_end_effector_index = 6
        self.kuka_gripper_index = 7

        # Null space parameters
        self.ll = [-0.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]  # Lower limits
        self.ul = [0.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]       # Upper limits
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]                   # Joint ranges
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -0.33 * math.pi, 0]  # Rest poses
        self.jd = [0.00001] * 14                                # Joint damping coefficients
        self.kuka_uid = None
        self.joint_positions = []
        self.num_joints = 0
        self.tray_uid = None
        self.end_effector_pos = []
        self.end_effector_angle = 0
        self.motor_names = []
        self.motor_indices = []

        self.reset()

    def reset(self):
        """Reset the robot and simulation environment."""
        objects = p.loadSDF(os.path.join(self.urdf_root_path, "kuka_iiwa/kuka_with_gripper2.sdf"))
        self.kuka_uid = objects[0]
        p.resetBasePositionAndOrientation(self.kuka_uid, [-0.1, 0, 0.07], [0, 0, 0, 1])

        self.joint_positions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539,
            0.000048, -0.299912, 0.0, -0.000043, 0.299960, 0.0, -0.000200
        ]
        self.num_joints = p.getNumJoints(self.kuka_uid)

        for joint_index in range(self.num_joints):
            p.resetJointState(self.kuka_uid, joint_index, self.joint_positions[joint_index])
            p.setJointMotorControl2(
                self.kuka_uid,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=self.joint_positions[joint_index],
                force=self.max_force,
            )

        self.tray_uid = p.loadURDF(
            os.path.join(self.urdf_root_path, "tray/tray.urdf"), 0.64, 0.075, -0.19, 0, 0, 1, 0
        )
        self.end_effector_pos = [0.537, 0.0, 0.5]
        self.end_effector_angle = 0

        self.motor_names = []
        self.motor_indices = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.kuka_uid, i)
            if joint_info[3] > -1:
                self.motor_names.append(str(joint_info[1]))
                self.motor_indices.append(i)

    def get_action_dimension(self):
        """Return the dimension of the action space."""
        if self.use_inverse_kinematics:
            return len(self.motor_indices)
        return 6  # x, y, z and roll/pitch/yaw of the end effector

    def get_observation_dimension(self):
        """Return the dimension of the observation space."""
        return len(self.get_observation())

    def get_observation(self):
        """Get the current state of the robot."""
        observation = []
        state = p.getLinkState(self.kuka_uid, self.kuka_gripper_index)
        pos, orn = state[0], state[1]
        euler = p.getEulerFromQuaternion(orn)
        observation.extend(pos)
        observation.extend(euler)
        return observation

    def apply_action(self, motor_commands):
        """Apply the given action to the robot."""
        if self.use_inverse_kinematics:
            self._apply_ik_action(motor_commands)
        else:
            for action_idx in range(len(motor_commands)):
                motor = self.motor_indices[action_idx]
                p.setJointMotorControl2(
                    self.kuka_uid,
                    motor,
                    p.POSITION_CONTROL,
                    targetPosition=motor_commands[action_idx],
                    force=self.max_force,
                )

    def _apply_ik_action(self, motor_commands):
        """Apply action using inverse kinematics."""
        dx, dy, dz, da, finger_angle = motor_commands

        self.end_effector_pos[0] = self._clamp(self.end_effector_pos[0] + dx, 0.50, 0.65)
        self.end_effector_pos[1] = self._clamp(self.end_effector_pos[1] + dy, -0.17, 0.22)
        self.end_effector_pos[2] += dz
        self.end_effector_angle += da

        pos = self.end_effector_pos
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])

        if self.use_null_space:
            joint_poses = p.calculateInverseKinematics(
                self.kuka_uid,
                self.kuka_end_effector_index,
                pos,
                orn,
                self.ll,
                self.ul,
                self.jr,
                self.rp,
            )
        else:
            joint_poses = p.calculateInverseKinematics(
                self.kuka_uid, self.kuka_end_effector_index, pos, orn, jointDamping=self.jd
            )

        for i in range(self.kuka_end_effector_index + 1):
            p.setJointMotorControl2(
                self.kuka_uid,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=self.max_force,
                maxVelocity=self.max_velocity,
                positionGain=0.3,
                velocityGain=1,
            )

        self._control_gripper(finger_angle)

    def _control_gripper(self, finger_angle):
        """Control the gripper."""
        p.setJointMotorControl2(
            self.kuka_uid, 7, p.POSITION_CONTROL, targetPosition=self.end_effector_angle, force=self.max_force
        )
        p.setJointMotorControl2(
            self.kuka_uid, 8, p.POSITION_CONTROL, targetPosition=-finger_angle, force=self.finger_a_force
        )
        p.setJointMotorControl2(
            self.kuka_uid, 11, p.POSITION_CONTROL, targetPosition=finger_angle, force=self.finger_b_force
        )
        p.setJointMotorControl2(self.kuka_uid, 10, p.POSITION_CONTROL, targetPosition=0, force=self.finger_tip_force)
        p.setJointMotorControl2(self.kuka_uid, 13, p.POSITION_CONTROL, targetPosition=0, force=self.finger_tip_force)

    @staticmethod
    def _clamp(value, min_value, max_value):
        """Clamp a value between min_value and max_value."""
        return max(min_value, min(value, max_value))

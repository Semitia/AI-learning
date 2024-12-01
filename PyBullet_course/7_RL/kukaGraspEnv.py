import os
import kuka
import random
import time
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from gym.utils import seeding
from pkg_resources import parse_version
import gym

# Constants
URDF_ROOT = pybullet_data.getDataPath()
LARGE_VAL_OBSERVATION = 100
RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class KukaGraspEnv(gym.Env):
    def __init__(self, urdf_root=URDF_ROOT, action_repeat=1, enable_self_collision=True,
                 renders=False, is_discrete=False, max_steps=1000):
        self._is_discrete = is_discrete
        self._time_step = 1. / 240.
        self._urdf_root = urdf_root
        self._action_repeat = action_repeat
        self._enable_self_collision = enable_self_collision
        self._observation = []
        self._env_step_counter = 0
        self._renders = renders
        self._max_steps = max_steps
        self.terminated = False
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40

        self._p = p
        self._initialize_simulation()

        self.seed()
        self.reset()
        observation_dim = len(self.get_extended_observation())
        observation_high = np.array([LARGE_VAL_OBSERVATION] * observation_dim)

        if self._is_discrete:
            self.action_space = spaces.Discrete(7)
        else:
            action_dim = 3
            action_bound = 1
            action_high = np.array([action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)

        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    def _initialize_simulation(self):
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)

    def reset(self):
        self.terminated = False
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._time_step)
        p.loadURDF(os.path.join(self._urdf_root, "plane.urdf"), [0, 0, -1])
        p.loadURDF(os.path.join(self._urdf_root, "table/table.urdf"), 0.5, 0.0, -0.82, 0, 0, 0, 1)

        xpos = 0.55 + 0.12 * random.random()
        ypos = 0.2 * random.random()
        ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])
        self.block_uid = p.loadURDF(os.path.join(self._urdf_root, "block.urdf"), xpos, ypos, -0.15, *orn)

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(urdfRootPath=self._urdf_root, timeStep=self._time_step)
        self._env_step_counter = 0
        p.stepSimulation()
        self._observation = self.get_extended_observation()
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_extended_observation(self):
        self._observation = self._kuka.getObservation()
        gripper_state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
        gripper_pos, gripper_orn = gripper_state[0], gripper_state[1]
        block_pos, block_orn = p.getBasePositionAndOrientation(self.block_uid)

        inv_gripper_pos, inv_gripper_orn = p.invertTransform(gripper_pos, gripper_orn)
        block_in_gripper_pos, block_in_gripper_orn = p.multiplyTransforms(
            inv_gripper_pos, inv_gripper_orn, block_pos, block_orn
        )
        block_euler_in_gripper = p.getEulerFromQuaternion(block_in_gripper_orn)
        block_in_gripper_pos_xy_eul_z = [block_in_gripper_pos[0], block_in_gripper_pos[1], block_euler_in_gripper[2]]

        self._observation.extend(list(block_in_gripper_pos_xy_eul_z))
        return self._observation

    def step(self, action):
        real_action = self._compute_action(action)
        return self._step_internal(real_action)

    def _compute_action(self, action):
        dv = 0.005
        if self._is_discrete:
            dx = [0, -dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0][action]
            da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
        else:
            dx, dy, da = action[0] * dv, action[1] * dv, action[2] * 0.05
        return [dx, dy, -0.002, da, 0.3]

    def _step_internal(self, action):
        for _ in range(self._action_repeat):
            self._kuka.applyAction(action)
            p.stepSimulation()
            if self._check_termination():
                break
            self._env_step_counter += 1

        if self._renders:
            time.sleep(self._time_step)

        self._observation = self.get_extended_observation()
        done = self._check_termination()
        reward = self._compute_reward(action)
        return np.array(self._observation), reward, done, {}

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._kuka.kukaUid)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1,
            farVal=100.0,
        )
        _, _, px, _, _ = self._p.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb_array = np.array(px, dtype=np.uint8).reshape(RENDER_HEIGHT, RENDER_WIDTH, 4)[:, :, :3]
        return rgb_array

    def _check_termination(self):
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        actual_end_effector_pos = state[0]

        if self.terminated or self._env_step_counter > self._max_steps:
            self._observation = self.get_extended_observation()
            return True

        closest_points = p.getClosestPoints(self._kuka.trayUid, self._kuka.kukaUid, 0.005)
        if closest_points:
            self.terminated = True
            self._close_gripper()
            return True
        return False

    def _close_gripper(self):
        finger_angle = 0.3
        for _ in range(100):
            grasp_action = [0, 0, 0.0001, 0, finger_angle]
            self._kuka.applyAction(grasp_action)
            p.stepSimulation()
            finger_angle -= 0.3 / 100.0
            finger_angle = max(finger_angle, 0)

    def _compute_reward(self, action):
        block_pos, _ = p.getBasePositionAndOrientation(self.block_uid)
        closest_points = p.getClosestPoints(self.block_uid, self._kuka.kukaUid, 1000, -1,
                                            self._kuka.kukaEndEffectorIndex)

        reward = -1000
        if closest_points:
            reward = -closest_points[0][8] * 10
        if block_pos[2] > 0.2:
            reward += 10000
        return reward


# Backward compatibility for older Gym versions
if parse_version(gym.__version__) < parse_version('0.9.6'):
    KukaGymEnv._render = KukaGymEnv.render
    KukaGymEnv._reset = KukaGymEnv.reset
    KukaGymEnv._seed = KukaGymEnv.seed
    KukaGymEnv._step = KukaGymEnv.step

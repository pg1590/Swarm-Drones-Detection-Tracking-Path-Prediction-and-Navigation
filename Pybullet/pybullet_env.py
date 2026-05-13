# pybullet_env.py

import pybullet as p
import pybullet_data
import numpy as np
from collections import deque


class PyBulletPursuitEnv:

    def __init__(self, gui=True):

        # ------------------------------------------------
        # Physics
        # ------------------------------------------------

        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(0, 0, -9.81)

        p.setTimeStep(1 / 60)

        # ------------------------------------------------
        # Environment params
        # ------------------------------------------------

        self.world_limit = 10

        self.capture_radius = 0.8

        self.max_steps = 300

        self.max_speed = 3.0

        self.history_len = 4

        # ------------------------------------------------
        # Agent placeholders
        # ------------------------------------------------

        self.p1 = None
        self.p2 = None
        self.evader = None

        # ------------------------------------------------
        # Histories
        # ------------------------------------------------

        self.p1_history = deque(
            maxlen=self.history_len
        )

        self.p2_history = deque(
            maxlen=self.history_len
        )

        self.reset()

    # ====================================================
    # RESET
    # ====================================================

    def reset(self):

        p.resetSimulation()

        p.setGravity(0, 0, -9.81)

        # ------------------------------------------------
        # Ground
        # ------------------------------------------------

        p.loadURDF("plane.urdf")

        # ------------------------------------------------
        # Visual shapes
        # ------------------------------------------------

        radius = 0.25

        mass = 1.0

        col_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius
        )

        # Pursuer 1
        vis1 = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[1, 0, 0, 1]
        )

        # Pursuer 2
        vis2 = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[0, 0, 1, 1]
        )

        # Evader
        vis3 = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[0, 1, 0, 1]
        )

        noise = 1.0

        self.p1_pos = np.array([
            -4,
            0,
            0.5
        ]) + noise * np.random.randn(3)

        self.p2_pos = np.array([
            4,
            0,
            0.5
        ]) + noise * np.random.randn(3)

        self.evader_pos = np.array([
            0,
            0,
            0.5
        ]) + noise * np.random.randn(3)

        self.p1 = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis1,
            basePosition=self.p1_pos
        )

        self.p2 = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis2,
            basePosition=self.p2_pos
        )

        self.evader = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis3,
            basePosition=self.evader_pos
        )

        # ------------------------------------------------
        # Velocity init
        # ------------------------------------------------

        self.p1_vel = np.zeros(3)

        self.p2_vel = np.zeros(3)

        self.evader_vel = np.zeros(3)

        self.step_count = 0

        self.capture = False

        # ------------------------------------------------
        # Histories
        # ------------------------------------------------

        obs1 = self._get_agent_obs(0)

        obs2 = self._get_agent_obs(1)

        for _ in range(self.history_len):

            self.p1_history.append(obs1)

            self.p2_history.append(obs2)

        return self.get_observations()

    # ====================================================
    # OBSERVATIONS
    # ====================================================

    def _get_agent_obs(self, agent_id):

        if agent_id == 0:

            self_pos = self.p1_pos
            self_vel = self.p1_vel
            other_pos = self.p2_pos

            one_hot = np.array([1, 0])

        else:

            self_pos = self.p2_pos
            self_vel = self.p2_vel
            other_pos = self.p1_pos

            one_hot = np.array([0, 1])

        relative_target_pos = (
            self.evader_pos - self_pos
        ) / self.world_limit

        relative_target_vel = (
            self.evader_vel - self_vel
        ) / self.max_speed

        relative_teammate = (
            other_pos - self_pos
        ) / self.world_limit

        obs = np.concatenate([

            relative_target_pos,

            relative_target_vel,

            self_vel / self.max_speed,

            relative_teammate,

            one_hot

        ])

        return obs.astype(np.float32)

    def get_observations(self):

        obs1 = self._get_agent_obs(0)

        obs2 = self._get_agent_obs(1)

        self.p1_history.append(obs1)

        self.p2_history.append(obs2)

        return {

            "p1": np.concatenate(
                list(self.p1_history)
            ),

            "p2": np.concatenate(
                list(self.p2_history)
            )
        }

    # ====================================================
    # GLOBAL STATE
    # ====================================================

    def get_global_state(self):

        return np.concatenate([

            self.p1_pos,

            self.p2_pos,

            self.evader_pos,

            self.p1_vel,

            self.p2_vel,

            self.evader_vel

        ]).astype(np.float32)

    # ====================================================
    # STEP
    # ====================================================

    def step(self, action1, action2):

        self.step_count += 1

        # ------------------------------------------------
        # Clip actions
        # ------------------------------------------------

        action1 = np.clip(
            action1,
            -1,
            1
        )

        action2 = np.clip(
            action2,
            -1,
            1
        )

        # ------------------------------------------------
        # Velocity control
        # ------------------------------------------------

        self.p1_vel = np.array([
            action1[0],
            action1[1],
            0
        ]) * self.max_speed

        self.p2_vel = np.array([
            action2[0],
            action2[1],
            0
        ]) * self.max_speed

        p.resetBaseVelocity(
            self.p1,
            linearVelocity=self.p1_vel
        )

        p.resetBaseVelocity(
            self.p2,
            linearVelocity=self.p2_vel
        )

        # ------------------------------------------------
        # Evader behavior
        # ------------------------------------------------

        vec1 = self.evader_pos - self.p1_pos
        vec2 = self.evader_pos - self.p2_pos

        d1 = np.linalg.norm(vec1) + 1e-6
        d2 = np.linalg.norm(vec2) + 1e-6

        dir1 = vec1 / d1
        dir2 = vec2 / d2

        if d1 < d2:

            desired_dir = (
                0.7 * dir1
                +
                0.3 * dir2
            )

        else:

            desired_dir = (
                0.7 * dir2
                +
                0.3 * dir1
            )

        desired_dir += (
            0.1 * np.random.randn(3)
        )

        desired_dir[2] = 0

        desired_dir /= (
            np.linalg.norm(desired_dir)
            + 1e-6
        )

        target_vel = desired_dir * 1.8

        self.evader_vel = (

            0.85 * self.evader_vel

            +

            0.15 * target_vel
        )

        p.resetBaseVelocity(
            self.evader,
            linearVelocity=self.evader_vel
        )

        # ------------------------------------------------
        # Physics step
        # ------------------------------------------------

        p.stepSimulation()

        # ------------------------------------------------
        # Read positions
        # ------------------------------------------------

        self.p1_pos = np.array(
            p.getBasePositionAndOrientation(
                self.p1
            )[0]
        )

        self.p2_pos = np.array(
            p.getBasePositionAndOrientation(
                self.p2
            )[0]
        )

        self.evader_pos = np.array(
            p.getBasePositionAndOrientation(
                self.evader
            )[0]
        )

        # ------------------------------------------------
        # Rewards
        # ------------------------------------------------

        reward, done = self.compute_rewards()

        observations = self.get_observations()

        return observations, reward, done

    # ====================================================
    # REWARD
    # ====================================================

    def compute_rewards(self):

        d1 = np.linalg.norm(
            self.p1_pos - self.evader_pos
        )

        d2 = np.linalg.norm(
            self.p2_pos - self.evader_pos
        )

        reward = 0.0

        reward -= 0.03 * (d1 + d2)

        # enclosure geometry
        v1 = self.p1_pos - self.evader_pos
        v2 = self.p2_pos - self.evader_pos

        v1 /= np.linalg.norm(v1) + 1e-6
        v2 /= np.linalg.norm(v2) + 1e-6

        dot = np.clip(
            np.dot(v1, v2),
            -1,
            1
        )

        angle = np.arccos(dot) / np.pi

        if d1 < 8 and d2 < 8:

            reward += 8.0 * angle

        done = False

        if (
            d1 < self.capture_radius
            or
            d2 < self.capture_radius
        ):

            reward += 1500

            self.capture = True

            print("CAPTURE!")

            done = True

        if self.step_count >= self.max_steps:

            done = True

        return reward, done
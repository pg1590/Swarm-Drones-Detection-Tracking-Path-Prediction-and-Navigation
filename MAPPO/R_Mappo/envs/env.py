import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class DroneSwarmEnv(gym.Env):

    def __init__(
        self,
        num_drones=3,
        gui=True,
        max_steps=500
    ):

        super(DroneSwarmEnv, self).__init__()

        self.num_drones = num_drones

        self.max_steps = max_steps

        self.current_step = 0

        self.gui = gui

        # =====================================================
        # PYBULLET
        # =====================================================

        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(0, 0, -9.8)

        # =====================================================
        # ACTION SPACE
        # =====================================================

        # vx, vy
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )

        # =====================================================
        # OBSERVATION SPACE
        # =====================================================

        # own pos (2)
        # own vel (2)
        # target relative pos (2)
        # neighbors relative pos (2*num_neighbors)

        obs_dim = (
            2
            + 2
            + 2
            + (self.num_drones - 1) * 2
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # =====================================================
        # LOAD ENVIRONMENT
        # =====================================================

        self.reset()

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        p.resetSimulation()

        p.setGravity(0, 0, -9.8)

        p.loadURDF("plane.urdf")

        self.current_step = 0

        # =====================================================
        # TARGET
        # =====================================================

        self.target_pos = np.array([
            np.random.uniform(-4, 4),
            np.random.uniform(-4, 4)
        ])

        # =====================================================
        # DRONES
        # =====================================================

        self.drones = []

        self.drone_positions = []

        self.drone_velocities = []

        for i in range(self.num_drones):

            start_pos = [
                np.random.uniform(-2, 2),
                np.random.uniform(-2, 2),
                0.2
            ]

            drone = p.loadURDF(
                "sphere2.urdf",
                start_pos,
                globalScaling=0.2
            )

            self.drones.append(drone)

        # =====================================================
        # TRAJECTORY HISTORY
        # =====================================================

        self.trajectory_history = []

        return self._get_obs()

    # =========================================================
    # STEP
    # =========================================================

    def step(self, actions):

        self.current_step += 1

        rewards = []

        dones = []

        next_obs = []

        positions = []

        # =====================================================
        # APPLY ACTIONS
        # =====================================================

        for i, drone in enumerate(self.drones):

            action = actions[i]

            vx = float(action[0])
            vy = float(action[1])

            pos, orn = p.getBasePositionAndOrientation(drone)

            new_pos = [
                pos[0] + vx * 0.05,
                pos[1] + vy * 0.05,
                0.2
            ]

            p.resetBasePositionAndOrientation(
                drone,
                new_pos,
                orn
            )

        p.stepSimulation()

        # =====================================================
        # OBS + REWARD
        # =====================================================

        for i, drone in enumerate(self.drones):

            pos, _ = p.getBasePositionAndOrientation(drone)

            vel, _ = p.getBaseVelocity(drone)

            pos = np.array(pos[:2])
            vel = np.array(vel[:2])

            positions.append(pos)

            # =================================================
            # TARGET DISTANCE
            # =================================================

            target_distance = np.linalg.norm(
                pos - self.target_pos
            )

            reward = -target_distance

            # =================================================
            # COLLISION PENALTY
            # =================================================

            for j, other_drone in enumerate(self.drones):

                if i == j:
                    continue

                other_pos, _ = p.getBasePositionAndOrientation(
                    other_drone
                )

                other_pos = np.array(other_pos[:2])

                dist = np.linalg.norm(pos - other_pos)

                if dist < 0.3:

                    reward -= 5.0

            # =================================================
            # TARGET BONUS
            # =================================================

            if target_distance < 0.5:

                reward += 20.0

            done = False

            if self.current_step >= self.max_steps:

                done = True

            rewards.append(reward)

            dones.append(done)

        # =====================================================
        # STORE TRAJECTORIES
        # =====================================================

        self.trajectory_history.append(
            np.array(positions)
        )

        next_obs = self._get_obs()

        return (
            next_obs,
            rewards,
            dones,
            {}
        )

    # =========================================================
    # OBSERVATIONS
    # =========================================================

    def _get_obs(self):

        obs_all = []

        for i, drone in enumerate(self.drones):

            pos, _ = p.getBasePositionAndOrientation(drone)

            vel, _ = p.getBaseVelocity(drone)

            pos = np.array(pos[:2])

            vel = np.array(vel[:2])

            obs = []

            # =================================================
            # OWN STATE
            # =================================================

            obs.extend(pos)

            obs.extend(vel)

            # =================================================
            # TARGET RELATIVE POSITION
            # =================================================

            rel_target = self.target_pos - pos

            obs.extend(rel_target)

            # =================================================
            # NEIGHBOR RELATIVE POSITIONS
            # =================================================

            for j, other_drone in enumerate(self.drones):

                if i == j:
                    continue

                other_pos, _ = p.getBasePositionAndOrientation(
                    other_drone
                )

                other_pos = np.array(other_pos[:2])

                rel_pos = other_pos - pos

                obs.extend(rel_pos)

            obs_all.append(
                np.array(obs, dtype=np.float32)
            )

        return obs_all

    # =========================================================
    # GLOBAL STATE
    # =========================================================

    def get_global_state(self):

        state = []

        for drone in self.drones:

            pos, _ = p.getBasePositionAndOrientation(
                drone
            )

            vel, _ = p.getBaseVelocity(drone)

            state.extend(pos[:2])

            state.extend(vel[:2])

        state.extend(self.target_pos)

        return np.array(state, dtype=np.float32)

    # =========================================================
    # RENDER
    # =========================================================

    def render(self):

        pass

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        p.disconnect()
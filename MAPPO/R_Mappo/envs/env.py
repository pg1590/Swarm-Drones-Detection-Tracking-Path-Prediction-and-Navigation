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

        self.angular_reward_scale = 3.0
        self.intercept_reward_scale = 3.0
        self.escape_block_reward_scale = 2.5
        self.parallel_penalty_scale = 1.5
        self.debug_stats = {
            "capture_rate": [],
            "mean_distance": [],
            "min_distance": [],
            "coverage": [],
        }
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
            2      # own pos
            + 2    # own vel
            + 2    # relative target pos
            + 2    # target velocity
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

        self.target_velocity = np.array([
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.05, 0.05)
        ])
        self.target_speed = 0.045
        # =====================================================
        # DRONES
        # =====================================================

        self.drones = []

        self.drone_positions = []
        self.prev_distances = np.zeros(self.num_drones)
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

        for i, drone in enumerate(self.drones):
            pos, _ = p.getBasePositionAndOrientation(drone)
            pos = np.array(pos[:2])

            self.prev_distances[i] = np.linalg.norm(
                pos - self.target_pos
            )
        # =====================================================
        # TRAJECTORY HISTORY
        # =====================================================

        self.trajectory_history = []
        self.target_history = []

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
        # EVASIVE TARGET POLICY
        # =====================================================

        drone_positions_temp = []

        for drone in self.drones:

            pos, _ = p.getBasePositionAndOrientation(drone)

            drone_positions_temp.append(
                np.array(pos[:2])
            )

        swarm_center = np.mean(
            np.array(drone_positions_temp),
            axis=0
        )

        escape_vec = (
            self.target_pos - swarm_center
        )

        escape_norm = np.linalg.norm(
            escape_vec
        )

        if escape_norm > 1e-5:

            escape_vec /= escape_norm

        noise = np.random.randn(2) * 0.2

        target_motion = (
            0.65 * escape_vec
            + 0.35 * noise
        )

        target_motion /= (
            np.linalg.norm(target_motion)
            + 1e-8
        )

        self.target_velocity = (
            target_motion
            * self.target_speed
        )

        self.target_pos += self.target_velocity

        # boundary reflection
        for k in range(2):

            if self.target_pos[k] > 8:
                self.target_velocity[k] *= -1

            if self.target_pos[k] < -8:
                self.target_velocity[k] *= -1

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

            # =====================================
            # PROGRESS REWARD
            # =====================================

            progress_reward = (
                self.prev_distances[i]
                - target_distance
            )

            # =====================================
            # DISTANCE REWARD
            # =====================================

            distance_reward = -0.05 * target_distance

            reward = (
                20.0 * progress_reward
                + distance_reward
            )

            # =====================================
            # VELOCITY ALIGNMENT
            # =====================================

            to_target = self.target_pos - pos

            norm = np.linalg.norm(to_target)

            if norm > 1e-5:

                to_target /= norm

                vel_norm = np.linalg.norm(vel)

                if vel_norm > 1e-5:

                    vel_dir = vel / vel_norm

                    alignment = np.dot(
                        vel_dir,
                        to_target
                    )

                    reward += 2.0 * alignment

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

                    reward -= 15.0

            # =================================================
            # COHESION REWARD
            # =================================================

            neighbor_distances = []

            for j, other_drone in enumerate(self.drones):

                if i == j:
                    continue

                other_pos, _ = p.getBasePositionAndOrientation(
                    other_drone
                )

                other_pos = np.array(other_pos[:2])

                dist = np.linalg.norm(pos - other_pos)

                neighbor_distances.append(dist)

            avg_neighbor_dist = np.mean(neighbor_distances)

            # desired swarm spacing
            if 1.0 < avg_neighbor_dist < 4.0:

                reward += 1.0

            # =================================================
            # ENCIRCLEMENT REWARD
            # =================================================

            angles = []

            for j, other_drone in enumerate(self.drones):

                if i == j:
                    continue

                other_pos, _ = p.getBasePositionAndOrientation(
                    other_drone
                )

                other_pos = np.array(other_pos[:2])

                vec = other_pos - self.target_pos

                angle = np.arctan2(vec[1], vec[0])

                angles.append(angle)

            if len(angles) >= 2:

                angles = np.sort(angles)

                angular_spread = angles[-1] - angles[0]

                reward += 0.5 * angular_spread
            

            # =================================================
            # ESCAPE CORRIDOR SUPPRESSION
            # =================================================

            all_angles = []

            for drone_id in self.drones:

                dpos, _ = p.getBasePositionAndOrientation(
                    drone_id
                )

                dpos = np.array(dpos[:2])

                vec = dpos - self.target_pos

                angle = np.arctan2(vec[1], vec[0])

                all_angles.append(angle)

            all_angles = np.sort(all_angles)

            largest_gap = 0

            for k in range(len(all_angles) - 1):

                gap = all_angles[k + 1] - all_angles[k]

                largest_gap = max(largest_gap, gap)

            # circular wraparound gap
            wrap_gap = (
                2 * np.pi
                - all_angles[-1]
                + all_angles[0]
            )

            largest_gap = max(largest_gap, wrap_gap)

            # smaller gap = better enclosure
            reward += (2 * np.pi - largest_gap)


            # =================================================
            # TARGET BONUS
            # =================================================

            if target_distance < 1.0:

                reward += 30.0

            done = False

            if self.current_step >= self.max_steps:

                done = True

            self.prev_distances[i] = target_distance
            rewards.append(reward)
            capture = False

            if target_distance < 0.8:
                reward += 100.0
                capture = True

            dones.append(done)

        if capture:
            dones = [True] * self.num_drones
        # =====================================================
        # STORE TRAJECTORIES
        # =====================================================

        self.trajectory_history.append(
            np.array(positions)
        )
        self.target_history.append(
            self.target_pos.copy()
        )

        # =====================================================
        # GLOBAL SWARM REWARDS
        # =====================================================

        # ---------------------------------
        # ANGULAR COVERAGE
        # ---------------------------------

        angles = []

        for pos in positions:

            dx = pos[0] - self.target_pos[0]
            dy = pos[1] - self.target_pos[1]

            angle = np.arctan2(dy, dx)

            angles.append(angle)

        angles = np.sort(np.array(angles))

        angular_gaps = []

        for i in range(len(angles)):

            next_i = (i + 1) % len(angles)

            gap = angles[next_i] - angles[i]

            if gap < 0:
                gap += 2 * np.pi

            angular_gaps.append(gap)

        largest_gap = max(angular_gaps)

        coverage_reward = (
            (2 * np.pi - largest_gap)
            / (2 * np.pi)
        )

        # ---------------------------------
        # INTERCEPTION REWARD
        # ---------------------------------

        target_speed = np.linalg.norm(
            self.target_velocity
        )

        intercept_reward = 0.0

        future_target = (
            self.target_pos
            + 8.0 * self.target_velocity
        )

        for pos in positions:

            future_distance = np.linalg.norm(
                pos - future_target
            )

            intercept_reward += (
                -future_distance
            )

        # ---------------------------------
        # PARALLEL CHASING PENALTY
        # ---------------------------------

        parallel_penalty = 0.0

        velocities = []

        for drone in self.drones:

            vel, _ = p.getBaseVelocity(drone)

            velocities.append(
                np.array(vel[:2])
            )

        for i in range(len(velocities)):

            for j in range(i + 1, len(velocities)):

                vi = velocities[i]
                vj = velocities[j]

                ni = np.linalg.norm(vi)
                nj = np.linalg.norm(vj)

                if ni > 1e-5 and nj > 1e-5:

                    cos_sim = (
                        np.dot(vi, vj)
                        / (ni * nj)
                    )

                    parallel_penalty += cos_sim

        # ---------------------------------
        # ESCAPE BLOCK REWARD
        # ---------------------------------

        escape_dir = -self.target_velocity

        escape_norm = np.linalg.norm(
            escape_dir
        )

        escape_block_reward = 0.0

        if escape_norm > 1e-5:

            escape_dir /= escape_norm

            for pos in positions:

                rel = pos - self.target_pos

                rel_norm = np.linalg.norm(rel)

                if rel_norm > 1e-5:

                    rel /= rel_norm

                    alignment = np.dot(
                        rel,
                        escape_dir
                    )

                    escape_block_reward += alignment

        # ---------------------------------
        # APPLY GLOBAL REWARDS
        # ---------------------------------

        # ---------------------------------
        # ROLE DIVERSITY REWARD
        # ---------------------------------

        diversity_reward = 0.0

        for i in range(len(positions)):

            for j in range(i + 1, len(positions)):

                rel_i = positions[i] - self.target_pos
                rel_j = positions[j] - self.target_pos

                ni = np.linalg.norm(rel_i)
                nj = np.linalg.norm(rel_j)

                if ni > 1e-5 and nj > 1e-5:

                    rel_i /= ni
                    rel_j /= nj

                    cosine = np.dot(rel_i, rel_j)

                    diversity_reward += (
                        1.0 - cosine
                    )

        global_reward = 0.0

        global_reward += (
            self.angular_reward_scale
            * coverage_reward
        )

        global_reward += (
            self.intercept_reward_scale
            * intercept_reward
        )

        global_reward += (
            self.escape_block_reward_scale
            * escape_block_reward
        )

        global_reward -= (
            self.parallel_penalty_scale
            * parallel_penalty
        )
        global_reward += 4.0 * diversity_reward
        rewards = [
            r + global_reward
            for r in rewards
        ]
        next_obs = self._get_obs()
        all_distances = []

        for pos in positions:

            d = np.linalg.norm(
                pos - self.target_pos
            )

            all_distances.append(d)

        mean_distance = np.mean(all_distances)

        min_distance = np.min(all_distances)

        capture_flag = (
            min_distance < 0.8
        )

        self.debug_stats["capture_rate"].append(
            int(capture_flag)
        )

        self.debug_stats["mean_distance"].append(
            mean_distance
        )

        self.debug_stats["min_distance"].append(
            min_distance
        )

        self.debug_stats["coverage"].append(
            coverage_reward
        )
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
            # TARGET VELOCITY
            # =================================================

            obs.extend(self.target_velocity)

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
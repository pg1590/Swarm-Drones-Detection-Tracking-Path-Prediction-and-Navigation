import gymnasium as gym
from gymnasium import spaces
import itertools
import numpy as np
import pybullet as p
import pybullet_data


class DroneSwarmEnv(gym.Env):

    def __init__(
        self,
        num_drones=3,
        gui=True,
        max_steps=300,
        control_dt=1.0 / 30.0,
        max_speed=2.0,
        velocity_response=0.35,
        capture_radius=0.8,
        safe_spacing=0.45
    ):

        super(DroneSwarmEnv, self).__init__()

        self.num_drones = num_drones
        self.max_steps = max_steps
        self.current_step = 0
        self.gui = gui

        self.control_dt = control_dt
        self.max_speed = max_speed
        self.velocity_response = velocity_response
        self.capture_radius = capture_radius
        self.safe_spacing = safe_spacing

        # Reward scales for Stage 1: homogeneous agents with emergent roles.
        self.pursuer_progress_scale = 8.0
        self.helper_progress_scale = 0.5
        self.team_progress_scale = 3.0
        self.closest_bonus_scale = 1.2
        self.coverage_scale = 2.0
        self.helper_support_scale = 3.0
        self.helper_ring_scale = 2.0
        self.helper_angle_scale = 2.0
        self.helper_alignment_scale = 0.4
        self.support_radius = 1.5
        self.support_angle = 2.0 * np.pi / 3.0
        self.escape_gap_target = 4.5
        self.helper_gap_penalty_scale = 3.0
        self.pursuer_gap_penalty_scale = 0.5
        self.capture_reward = 100.0
        self.capture_agent_bonus = 30.0
        self.max_shaping_reward = 30.0

        self.debug_stats = {
            "capture_rate": [],
            "mean_distance": [],
            "min_distance": [],
            "coverage": [],
            "escape_gap": [],
            "gap_excess": [],
            "helper_support_error": [],
            "helper_ring_error": [],
            "velocity_diversity": [],
        }

        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.control_dt)

        # vx, vy command in normalized units. Magnitude is meaningful.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        obs_dim = (
            2      # own position
            + 2    # own velocity
            + 2    # relative target position
            + 2    # target velocity
            + 2    # future target relative position
            + (self.num_drones - 1) * 4
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.reset()

    def reset(self):

        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.control_dt)
        p.loadURDF("plane.urdf")

        self.current_step = 0

        self.target_pos = np.array([
            np.random.uniform(-4, 4),
            np.random.uniform(-4, 4)
        ], dtype=np.float32)

        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.25,
            rgbaColor=[1, 0.1, 0.1, 1]
        )

        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.25
        )

        self.target_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[
                self.target_pos[0],
                self.target_pos[1],
                0.25
            ]
        )

        # Stage 1 target is stationary. Keep velocity in the state so the same
        # observation contract can support moving-target curriculum later.
        self.target_speed = 0.0
        self.target_velocity = np.array([0.0, 0.0], dtype=np.float32)

        self.drones = []
        self.prev_distances = np.zeros(self.num_drones, dtype=np.float32)

        for _ in range(self.num_drones):
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

            p.changeVisualShape(
                drone,
                -1,
                rgbaColor=[0.1, 0.4, 1, 1]
            )

            self.drones.append(drone)

        positions, _ = self._read_drone_states()
        distances = self._distances_to_target(positions)

        self.prev_distances = distances.copy()
        self.prev_team_distance = float(np.mean(distances))

        self.trajectory_history = []
        self.target_history = []

        return self._get_obs()

    def step(self, actions):

        self.current_step += 1

        processed_actions = []

        for i, drone in enumerate(self.drones):
            action = np.array(actions[i], dtype=np.float32)
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            action = np.clip(action, -1.0, 1.0)

            action_norm = np.linalg.norm(action)
            if action_norm > 1.0:
                action = action / action_norm

            current_vel, _ = p.getBaseVelocity(drone)
            current_vel = np.array(current_vel[:2], dtype=np.float32)

            desired_vel = action * self.max_speed
            new_vel = (
                (1.0 - self.velocity_response) * current_vel
                + self.velocity_response * desired_vel
            )

            speed = np.linalg.norm(new_vel)
            if speed > self.max_speed:
                new_vel = (new_vel / speed) * self.max_speed

            p.resetBaseVelocity(
                drone,
                linearVelocity=[
                    float(new_vel[0]),
                    float(new_vel[1]),
                    0.0
                ],
                angularVelocity=[0.0, 0.0, 0.0]
            )

            processed_actions.append(action)

        p.stepSimulation()

        self._update_target()

        positions, velocities = self._read_drone_states()
        distances = self._distances_to_target(positions)

        rewards, capture = self._compute_rewards(
            positions,
            velocities,
            distances,
            processed_actions
        )

        # Gymnasium-style split: capture is a true MDP terminal, while the
        # time limit and the escape cutoff end the episode early from a
        # recoverable state, so GAE must bootstrap their final states.
        terminated = bool(capture)

        truncated = self.current_step >= self.max_steps

        if float(np.mean(distances)) > 12.0:
            truncated = True

        terminateds = [terminated] * self.num_drones
        truncateds = [truncated] * self.num_drones

        self.prev_distances = distances.copy()
        self.prev_team_distance = float(np.mean(distances))

        self.trajectory_history.append(np.array(positions))
        self.target_history.append(self.target_pos.copy())

        next_obs = self._get_obs()

        return (
            next_obs,
            rewards,
            terminateds,
            truncateds,
            {
                "capture": bool(capture),
                "min_distance": float(np.min(distances)),
                "mean_distance": float(np.mean(distances)),
            }
        )

    def _update_target(self):

        self.target_pos = (
            self.target_pos
            + self.target_velocity * self.control_dt
        )

        for k in range(2):
            if self.target_pos[k] > 8:
                self.target_pos[k] = 8
                self.target_velocity[k] *= -1

            if self.target_pos[k] < -8:
                self.target_pos[k] = -8
                self.target_velocity[k] *= -1

        p.resetBasePositionAndOrientation(
            self.target_body,
            [
                float(self.target_pos[0]),
                float(self.target_pos[1]),
                0.25
            ],
            [0, 0, 0, 1]
        )

        if self.gui:
            p.addUserDebugLine(
                [
                    self.target_pos[0] - 1.0,
                    self.target_pos[1],
                    0.02
                ],
                [
                    self.target_pos[0] + 1.0,
                    self.target_pos[1],
                    0.02
                ],
                [1, 0, 0],
                lifeTime=0.05
            )

            p.addUserDebugLine(
                [
                    self.target_pos[0],
                    self.target_pos[1] - 1.0,
                    0.02
                ],
                [
                    self.target_pos[0],
                    self.target_pos[1] + 1.0,
                    0.02
                ],
                [1, 0, 0],
                lifeTime=0.05
            )

    def _compute_rewards(
        self,
        positions,
        velocities,
        distances,
        actions
    ):

        rewards = np.full(self.num_drones, -0.01, dtype=np.float32)

        closest_idx = int(np.argmin(distances))
        min_distance = float(distances[closest_idx])
        mean_distance = float(np.mean(distances))

        individual_progress = self.prev_distances - distances
        team_progress = self.prev_team_distance - mean_distance

        coverage_reward, largest_gap = self._angular_coverage(positions)
        coverage_gate = np.clip((5.0 - mean_distance) / 3.0, 0.0, 1.0)
        gap_excess = max(0.0, largest_gap - self.escape_gap_target)
        velocity_diversity = self._velocity_diversity(velocities)
        helper_targets = self._helper_support_assignments(
            positions,
            closest_idx
        )

        capture = min_distance < self.capture_radius
        helper_support_errors = []
        helper_ring_errors = []

        for i in range(self.num_drones):
            if i == closest_idx:
                rewards[i] += self.closest_bonus_scale / (distances[i] + 0.5)
                rewards[i] += (
                    self.pursuer_progress_scale
                    * individual_progress[i]
                )
                rewards[i] -= (
                    self.pursuer_gap_penalty_scale
                    * gap_excess
                )
            else:
                rewards[i] += (
                    self.helper_progress_scale
                    * individual_progress[i]
                )
                rewards[i] -= (
                    self.helper_gap_penalty_scale
                    * gap_excess
                )

            rewards[i] += self.team_progress_scale * team_progress
            rewards[i] += 0.8 / (mean_distance + 1.0)

            rewards[i] += (
                self.coverage_scale
                * coverage_gate
                * self._coverage_contribution(positions, i, coverage_reward)
            )

            to_target = self.target_pos - positions[i]
            to_target_norm = np.linalg.norm(to_target)
            speed = np.linalg.norm(velocities[i])

            if to_target_norm > 1e-6 and speed > 1e-6:
                target_dir = to_target / to_target_norm
                vel_dir = velocities[i] / speed
                alignment = float(np.dot(vel_dir, target_dir))

                if i == closest_idx:
                    rewards[i] += 0.6 * alignment
                else:
                    rewards[i] += 0.1 * alignment

            if i != closest_idx and i in helper_targets:
                support_reward, support_error, ring_error = \
                    self._helper_support_reward(
                        positions[i],
                        velocities[i],
                        helper_targets[i]
                    )

                rewards[i] += support_reward
                helper_support_errors.append(support_error)
                helper_ring_errors.append(ring_error)

            if speed < 0.05 and distances[i] > self.capture_radius:
                rewards[i] -= 0.05

            rewards[i] -= 0.02 * float(np.linalg.norm(actions[i]) ** 2)
            rewards[i] -= self._spacing_penalty(positions, i)

        self.debug_stats["capture_rate"].append(int(capture))
        self.debug_stats["mean_distance"].append(mean_distance)
        self.debug_stats["min_distance"].append(min_distance)
        self.debug_stats["coverage"].append(float(coverage_reward))
        self.debug_stats["escape_gap"].append(float(largest_gap))
        self.debug_stats["gap_excess"].append(float(gap_excess))
        self.debug_stats["helper_support_error"].append(
            float(np.mean(helper_support_errors))
            if len(helper_support_errors) > 0
            else 0.0
        )
        self.debug_stats["helper_ring_error"].append(
            float(np.mean(helper_ring_errors))
            if len(helper_ring_errors) > 0
            else 0.0
        )
        self.debug_stats["velocity_diversity"].append(float(velocity_diversity))

        rewards = np.clip(
            rewards,
            -self.max_shaping_reward,
            self.max_shaping_reward
        )

        if capture:
            rewards += self.capture_reward
            rewards[closest_idx] += self.capture_agent_bonus

        return [float(r) for r in rewards], capture

    def _helper_support_reward(
        self,
        position,
        velocity,
        support_target
    ):

        support_error = np.linalg.norm(position - support_target)

        support_score = np.exp(
            -(support_error ** 2) / 1.0
        )

        distance_to_target = np.linalg.norm(
            position - self.target_pos
        )

        ring_error = abs(
            distance_to_target
            - self.support_radius
        )

        ring_score = np.exp(
            -(ring_error ** 2) / 0.5
        )

        helper_rel = position - self.target_pos
        support_rel = support_target - self.target_pos
        helper_angle = np.arctan2(helper_rel[1], helper_rel[0])
        support_angle = np.arctan2(support_rel[1], support_rel[0])
        angular_error = abs(
            self._wrap_angle(helper_angle - support_angle)
        )

        angle_score = np.exp(
            -(angular_error ** 2) / 0.5
        )

        reward = (
            self.helper_support_scale * support_score
            + self.helper_ring_scale * ring_score
            + self.helper_angle_scale * angle_score
        )

        to_support = support_target - position
        to_support_norm = np.linalg.norm(to_support)
        speed = np.linalg.norm(velocity)

        if to_support_norm > 1e-6 and speed > 1e-6:
            support_dir = to_support / to_support_norm
            vel_dir = velocity / speed
            reward += self.helper_alignment_scale * float(
                np.dot(vel_dir, support_dir)
            )

        return reward, float(support_error), float(ring_error)

    def _wrap_angle(self, angle):

        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _helper_support_assignments(
        self,
        positions,
        closest_idx
    ):

        helper_indices = [
            i
            for i in range(self.num_drones)
            if i != closest_idx
        ]

        if len(helper_indices) == 0:
            return {}

        pursuer_rel = positions[closest_idx] - self.target_pos
        pursuer_norm = np.linalg.norm(pursuer_rel)

        if pursuer_norm > 1e-6:
            pursuer_angle = np.arctan2(
                pursuer_rel[1],
                pursuer_rel[0]
            )
        else:
            pursuer_angle = 0.0

        if len(helper_indices) == 1:
            support_offsets = [np.pi]
        else:
            support_offsets = np.linspace(
                -self.support_angle,
                self.support_angle,
                len(helper_indices)
            )

        support_points = []

        for offset in support_offsets:
            support_angle = pursuer_angle + offset
            support_points.append(
                self.target_pos
                + self.support_radius * np.array([
                    np.cos(support_angle),
                    np.sin(support_angle)
                ], dtype=np.float32)
            )

        best_assignment = None
        best_cost = np.inf

        for permutation in itertools.permutations(helper_indices):
            cost = 0.0

            for helper_idx, support_point in zip(
                permutation,
                support_points
            ):
                cost += np.linalg.norm(
                    positions[helper_idx]
                    - support_point
                )

            if cost < best_cost:
                best_cost = cost
                best_assignment = permutation

        return {
            helper_idx: support_point
            for helper_idx, support_point in zip(
                best_assignment,
                support_points
            )
        }

    def _spacing_penalty(self, positions, drone_idx):

        penalty = 0.0

        for j, other_pos in enumerate(positions):
            if drone_idx == j:
                continue

            dist = np.linalg.norm(positions[drone_idx] - other_pos)

            if dist < self.safe_spacing:
                penalty += 5.0 * (
                    self.safe_spacing - dist
                ) / self.safe_spacing

        return penalty

    def _angular_coverage(self, positions):

        if len(positions) < 2:
            return 0.0, 2 * np.pi

        angles = []

        for pos in positions:
            rel = pos - self.target_pos
            angles.append(np.arctan2(rel[1], rel[0]))

        angles = np.sort(np.array(angles))
        largest_gap = self._largest_angular_gap(angles)

        coverage_reward = (
            2 * np.pi - largest_gap
        ) / (2 * np.pi)

        return float(coverage_reward), float(largest_gap)

    def _coverage_contribution(
        self,
        positions,
        drone_idx,
        team_coverage
    ):

        if len(positions) <= 2:
            return team_coverage

        reduced_positions = [
            pos
            for i, pos in enumerate(positions)
            if i != drone_idx
        ]

        reduced_coverage, _ = self._angular_coverage(reduced_positions)

        return max(0.0, team_coverage - reduced_coverage)

    def _largest_angular_gap(self, angles):

        largest_gap = 0.0

        for i in range(len(angles)):
            next_i = (i + 1) % len(angles)
            gap = angles[next_i] - angles[i]

            if gap < 0:
                gap += 2 * np.pi

            largest_gap = max(largest_gap, gap)

        return largest_gap

    def _velocity_diversity(self, velocities):

        if len(velocities) < 2:
            return 0.0

        diversity = 0.0
        pairs = 0

        for i in range(len(velocities)):
            for j in range(i + 1, len(velocities)):
                diversity += np.linalg.norm(velocities[i] - velocities[j])
                pairs += 1

        return diversity / max(pairs, 1)

    def _read_drone_states(self):

        positions = []
        velocities = []

        for drone in self.drones:
            pos, _ = p.getBasePositionAndOrientation(drone)
            vel, _ = p.getBaseVelocity(drone)

            positions.append(np.array(pos[:2], dtype=np.float32))
            velocities.append(np.array(vel[:2], dtype=np.float32))

        return positions, velocities

    def _distances_to_target(self, positions):

        distances = [
            np.linalg.norm(pos - self.target_pos)
            for pos in positions
        ]

        return np.array(distances, dtype=np.float32)

    def _get_obs(self):

        obs_all = []

        for i, drone in enumerate(self.drones):
            pos, _ = p.getBasePositionAndOrientation(drone)
            vel, _ = p.getBaseVelocity(drone)

            pos = np.array(pos[:2], dtype=np.float32)
            vel = np.array(vel[:2], dtype=np.float32)

            obs = []
            obs.extend(pos)
            obs.extend(vel)

            rel_target = self.target_pos - pos
            obs.extend(rel_target)

            obs.extend(self.target_velocity)

            future_target = (
                self.target_pos
                + 4.0 * self.target_velocity * self.control_dt
            )

            future_rel = future_target - pos
            obs.extend(future_rel)

            for j, other_drone in enumerate(self.drones):
                if i == j:
                    continue

                other_pos, _ = p.getBasePositionAndOrientation(other_drone)
                other_vel, _ = p.getBaseVelocity(other_drone)

                other_pos = np.array(other_pos[:2], dtype=np.float32)
                other_vel = np.array(other_vel[:2], dtype=np.float32)

                obs.extend(other_pos - pos)
                obs.extend(other_vel - vel)

            obs_all.append(np.array(obs, dtype=np.float32))

        return obs_all

    def get_global_state(self, agent_idx):

        # Agent-centric ordering: own pos/vel first, then the remaining
        # drones in index order, then the target. All agents see the same
        # information, but the shared critic can tell whose (per-agent)
        # return it is predicting.
        positions, velocities = self._read_drone_states()

        order = [agent_idx] + [
            j
            for j in range(self.num_drones)
            if j != agent_idx
        ]

        state = []

        for j in order:
            state.extend(positions[j])
            state.extend(velocities[j])

        state.extend(self.target_pos)

        return np.array(state, dtype=np.float32)

    def render(self):
        pass

    def close(self):
        p.disconnect()

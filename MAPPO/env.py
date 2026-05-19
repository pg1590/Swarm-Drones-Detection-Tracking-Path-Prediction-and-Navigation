import numpy as np
from collections import deque
from utils import compute_swarm_metrics

class PursuitEnv:

    def __init__(self):

        self.dt = 0.1

        self.max_steps = 150

        self.capture_radius = 0.5

        self.world_limit = 20.0

        self.max_speed = 2.5

        # observation history length
        self.history_len = 3

        self.reset()

    def reset(self):

        # ------------------------------------------------
        # Structured asymmetric initialization
        # ------------------------------------------------

        noise_scale = 1.0

        self.p1_pos = np.array([
            -4.0,
            0.0
        ]) + noise_scale * np.random.randn(2)

        self.p2_pos = np.array([
            4.0,
            0.0
        ]) + noise_scale * np.random.randn(2)

        self.evader_pos = np.array([
            0.0,
            0.0
        ]) + noise_scale * np.random.randn(2)

        # ------------------------------------------------
        # Velocities
        # ------------------------------------------------

        self.p1_vel = np.zeros(2)

        self.p2_vel = np.zeros(2)

        self.evader_vel = np.zeros(2)

        self.step_count = 0

        self.capture = False
        self.enclosure_steps=0
        # ------------------------------------------------
        # Observation histories
        # ------------------------------------------------

        self.p1_history = deque(maxlen=self.history_len)

        self.p2_history = deque(maxlen=self.history_len)

        obs1 = self._get_agent_obs(agent_id=0)

        obs2 = self._get_agent_obs(agent_id=1)

        for _ in range(self.history_len):

            self.p1_history.append(obs1)

            self.p2_history.append(obs2)

        return self.get_observations()

    def step(self, a1, a2):

        # -----------------------------
        # Clip actions
        # -----------------------------

        a1 = np.clip(a1, -1.0, 1.0)
        a2 = np.clip(a2, -1.0, 1.0)

        self.p1_vel = a1 * self.max_speed
        self.p2_vel = a2 * self.max_speed
        
        # ------------------------------------------------
        # Smarter evasive target
        # ------------------------------------------------

        vec1 = self.evader_pos - self.p1_pos
        vec2 = self.evader_pos - self.p2_pos

        d1 = np.linalg.norm(vec1) + 1e-6
        d2 = np.linalg.norm(vec2) + 1e-6

        dir1 = vec1 / d1
        dir2 = vec2 / d2

        # ------------------------------------------------
        # Escape direction
        # ------------------------------------------------

        if d1 < d2:

            desired_direction = (
                0.7 * dir1
                +
                0.3 * dir2
            )

        else:

            desired_direction = (
                0.7 * dir2
                +
                0.3 * dir1
            )

        # ------------------------------------------------
        # Wall avoidance
        # ------------------------------------------------

        wall_repulsion = np.zeros(2)

        margin = 3.0

        for i in range(2):

            if self.evader_pos[i] > self.world_limit - margin:

                wall_repulsion[i] -= 1.0

            elif self.evader_pos[i] < -self.world_limit + margin:

                wall_repulsion[i] += 1.0

        desired_direction += 0.5 * wall_repulsion

        # ------------------------------------------------
        # Exploration noise
        # ------------------------------------------------

        noise = 0.1 * np.random.randn(2)

        desired_direction += noise

        desired_direction /= (
            np.linalg.norm(desired_direction)
            + 1e-6
        )

        # ------------------------------------------------
        # Momentum-constrained evader
        # ------------------------------------------------

        evader_speed = 1.8

        target_velocity = (
            desired_direction * evader_speed
        )

        # smooth dynamics
        self.evader_vel = (

            0.85 * self.evader_vel

            +

            0.15 * target_velocity
)
        # -----------------------------
        # Physics update
        # -----------------------------

        self.p1_pos += self.p1_vel * self.dt
        self.p2_pos += self.p2_vel * self.dt

        self.evader_pos += self.evader_vel * self.dt

        # -----------------------------
        # Bound world
        # -----------------------------

        self.p1_pos = np.clip(
            self.p1_pos,
            -self.world_limit,
            self.world_limit
        )

        self.p2_pos = np.clip(
            self.p2_pos,
            -self.world_limit,
            self.world_limit
        )

        self.evader_pos = np.clip(
            self.evader_pos,
            -self.world_limit,
            self.world_limit
        )

        self.step_count += 1

        # -----------------------------
        # Rewards
        # -----------------------------

        reward, done = self.compute_rewards()

        # -----------------------------
        # Update histories
        # -----------------------------

        self.p1_history.append(
            self._get_agent_obs(agent_id=0)
        )

        self.p2_history.append(
            self._get_agent_obs(agent_id=1)
        )

        return (
            self.get_observations(),
            reward,
            done
        )

    def compute_rewards(self):

        from utils import enclosure_reward

        # ------------------------------------------------
        # Distances to target
        # ------------------------------------------------

        d1 = np.linalg.norm(
            self.p1_pos - self.evader_pos
        )

        d2 = np.linalg.norm(
            self.p2_pos - self.evader_pos
        )

        # ------------------------------------------------
        # Future target prediction
        # ------------------------------------------------

        prediction_horizon = 0.6

        future_target = (

            self.evader_pos

            +

            prediction_horizon
            * self.evader_vel
        )

        d1_future = np.linalg.norm(
            self.p1_pos - future_target
        )

        d2_future = np.linalg.norm(
            self.p2_pos - future_target
        )

        # ------------------------------------------------
        # Inter-agent distance
        # ------------------------------------------------

        inter_agent_dist = np.linalg.norm(
            self.p1_pos - self.p2_pos
        )

        # ------------------------------------------------
        # Geometric enclosure reward
        # ------------------------------------------------

        angle_reward = enclosure_reward(

            self.p1_pos,

            self.p2_pos,

            self.evader_pos
        )
        metric=compute_swarm_metrics(
            self.p1_pos,

            self.p2_pos,

            self.evader_pos

        )

        # ------------------------------------------------
        # Reward initialization
        # ------------------------------------------------

        reward = 0.0

        # =================================================
        # 1. Future interception reward
        # =================================================

        reward -= 0.08 * (
            d1_future + d2_future
        )

        # =================================================
        # 2. Current pursuit reward
        # =================================================

        reward -= 0.03 * (
            d1 + d2
        )

        # =================================================
        # 3. Enclosure reward
        # =================================================

        if d1 < 8.0 and d2 < 8.0:

            reward += 8.0 * angle_reward
        
        if metric["enclosure_angle"] > 120:

            self.enclosure_steps += 1

            reward += 0.2

            reward += 0.01 * self.enclosure_steps

        else:

            self.enclosure_steps = 0
        # =================================================
        # 4. Prevent overlap collapse
        # =================================================

        if inter_agent_dist < 1.0:

            reward -= 15.0

        # =================================================
        # 5. Mild motion penalty
        # =================================================

        reward -= 0.001 * (

            np.linalg.norm(self.p1_vel)

            +

            np.linalg.norm(self.p2_vel)
        )

        # =================================================
        # 6. Wall penalty
        # =================================================

        wall_threshold = 2.0

        for pos in [

            self.p1_pos,

            self.p2_pos
        ]:

            dist_to_wall = (

                self.world_limit

                -

                np.max(np.abs(pos))
            )

            if dist_to_wall < wall_threshold:

                reward -= (

                    1.5

                    *

                    (wall_threshold - dist_to_wall)
                )

        # =================================================
        # 7. Capture condition
        # =================================================

        done = False

        if (

            d1 < self.capture_radius

            and

            d2 < self.capture_radius
        ):

            reward += 200.0

            self.capture = True

            print("CAPTURE!")

            done = True

        # =================================================
        # 8. Timeout
        # =================================================

        if self.step_count >= self.max_steps:

            reward -= 10.0

            done = True

        reward=reward/50.0
        return reward, done
    
    def _get_agent_obs(self, agent_id):

        if agent_id == 0:

            self_pos = self.p1_pos

            self_vel = self.p1_vel

            other_pos = self.p2_pos

            one_hot = np.array([1.0, 0.0])

        else:

            self_pos = self.p2_pos

            self_vel = self.p2_vel

            other_pos = self.p1_pos

            one_hot = np.array([0.0, 1.0])

        # ------------------------------------------------
        # Relative target information
        # ------------------------------------------------

        relative_target_pos = (

            self.evader_pos - self_pos

        ) / self.world_limit

        relative_target_vel = (

            self.evader_vel - self_vel

        ) / self.max_speed

        # ------------------------------------------------
        # Teammate geometry cue
        # ------------------------------------------------

        relative_teammate = (

            other_pos - self_pos

        ) / self.world_limit

        teammate_distance = np.array([

            np.linalg.norm(relative_teammate)

        ])

        # ------------------------------------------------
        # Wall features
        # ------------------------------------------------

        wall_features = np.array([

            (
                self.world_limit
                - abs(self_pos[0])
            ) / self.world_limit,

            (
                self.world_limit
                - abs(self_pos[1])
            ) / self.world_limit

        ])

        # ------------------------------------------------
        # Observation vector
        # ------------------------------------------------

        obs = np.concatenate([

            relative_target_pos,

            relative_target_vel,

            self_vel / self.max_speed,

            relative_teammate,

            teammate_distance,

            wall_features,

            one_hot

        ])

        return obs.astype(np.float32)

    def get_observations(self):

        obs1 = np.concatenate(
            list(self.p1_history)
        )

        obs2 = np.concatenate(
            list(self.p2_history)
        )

        return {
            "p1": obs1,
            "p2": obs2
        }

    def get_global_state(self):

        """
        Used ONLY by centralized critic
        """

        global_state = np.concatenate([

            self.p1_pos,
            self.p1_vel,

            self.p2_pos,
            self.p2_vel,

            self.evader_pos,
            self.evader_vel

        ])

        return global_state.astype(np.float32)
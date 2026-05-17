import numpy as np

class PursuitEnv:
    def __init__(self):

        self.dt = 1.0
        self.max_steps = 200
        self.capture_radius = 1.5
        self.world_limit = 10.0
        self.total_episodes = 0
        self.max_speed = 1.0

        self.reset()

    def reset(self):

        self.p1_pos = np.random.uniform(-4, 4, size=3)
        self.p2_pos = np.random.uniform(-4, 4, size=3)
        self.evader_pos = np.random.uniform(-4, 4, size=3)

        self.p1_vel = np.zeros(3)
        self.p2_vel = np.zeros(3)
        self.evader_vel = np.zeros(3)

        self.step_count = 0
        difficulty = min(1.0, self.total_episodes / 3000)

        return self.get_states()

    def step(self, a1, a2, a_e=None):

        # --- Clip pursuer velocities ---
        a1 = np.clip(a1, -self.max_speed, self.max_speed)
        a2 = np.clip(a2, -self.max_speed, self.max_speed)

        self.p1_vel = a1
        self.p2_vel = a2

        # --------------------------------
        # Curriculum Evader
        # --------------------------------

        difficulty = min(1.0, self.step_count / 5000)

        random_dir = np.random.randn(3)
        random_dir /= (np.linalg.norm(random_dir) + 1e-6)

        vec1 = self.evader_pos - self.p1_pos
        vec2 = self.evader_pos - self.p2_pos

        smart_dir = (
            vec1 / (np.linalg.norm(vec1) + 1e-6)
            +
            vec2 / (np.linalg.norm(vec2) + 1e-6)
        )

        smart_dir /= (np.linalg.norm(smart_dir) + 1e-6)

        direction = (
            (1 - difficulty) * random_dir
            +
            difficulty * smart_dir
        )

        direction /= (np.linalg.norm(direction) + 1e-6)

        # Add noise
        noise = 0.1 * np.random.randn(3)
        direction = direction + noise
        direction = direction / (np.linalg.norm(direction) + 1e-6)



        # Evader slightly slower than pursuers
        evader_speed = 1
        self.evader_vel = evader_speed * direction

        # --- Update positions ---
        self.p1_pos += self.p1_vel * self.dt
        self.p2_pos += self.p2_vel * self.dt
        self.evader_pos += self.evader_vel * self.dt

        # # --- Bound world ---
        self.p1_pos = np.clip(self.p1_pos, -self.world_limit, self.world_limit)
        self.p2_pos = np.clip(self.p2_pos, -self.world_limit, self.world_limit)
        self.evader_pos = np.clip(self.evader_pos, -self.world_limit, self.world_limit)

        self.step_count += 1

        rewards, done = self.compute_rewards()

        return self.get_states(), rewards, done

    def compute_rewards(self):

        d1 = np.linalg.norm(self.p1_pos - self.evader_pos)
        d2 = np.linalg.norm(self.p2_pos - self.evader_pos)

        # -----------------------------
        # 1. Distance shaping
        # -----------------------------
        r_dist = -0.05 * (d1 + d2)

        # -----------------------------
        # 2. Anti-clustering reward
        # Encourage pursuers to spread
        # -----------------------------
        pursuer_sep = np.linalg.norm(self.p1_pos - self.p2_pos)

        r_sep = 0.1 * np.clip(pursuer_sep, 0, 10)

        # -----------------------------
        # 3. Angular enclosure reward
        # Encourage target to lie
        # between pursuers
        # -----------------------------
        v1 = self.p1_pos - self.evader_pos
        v2 = self.p2_pos - self.evader_pos

        v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)

        dot = np.dot(v1_norm, v2_norm)

        # dot = -1 means 180 degrees
        r_angle = -0.5 * dot

        # -----------------------------
        # 4. Interception reward
        # Predict future target position
        # -----------------------------
        pred_target = self.evader_pos + 2.0 * self.evader_vel

        d1_pred = np.linalg.norm(self.p1_pos - pred_target)
        d2_pred = np.linalg.norm(self.p2_pos - pred_target)

        r_intercept = -0.03 * (d1_pred + d2_pred)

        # -----------------------------
        # Total reward
        # -----------------------------
        r = r_dist + r_sep + r_angle + r_intercept

        done = False

        # -----------------------------
        # Capture reward
        # -----------------------------
        if d1 < self.capture_radius or d2 < self.capture_radius:
            r += 20
            done = True

        # -----------------------------
        # Timeout
        # -----------------------------
        if self.step_count >= self.max_steps:
            done = True

        return (r, r), done


    def get_states(self):
        return {
            "p1": self.p1_pos.copy(),
            "p2": self.p2_pos.copy(),
            "evader": self.evader_pos.copy()
        }

import numpy as np

class PursuitEnv:
    def __init__(self):

        self.dt = 1.0
        self.max_steps = 200
        self.capture_radius = 0.2
        self.world_limit = 50.0

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

        return self.get_states()

    def step(self, a1, a2, a_e=None):

        # --- Clip pursuer velocities ---
        a1 = np.clip(a1, -self.max_speed, self.max_speed)
        a2 = np.clip(a2, -self.max_speed, self.max_speed)

        self.p1_vel = a1
        self.p2_vel = a2

        # --- Reactive evader ---
        # Find nearest pursuer
        # --- Smarter Evader: maximize distance to both pursuers ---

        vec1 = self.evader_pos - self.p1_pos
        vec2 = self.evader_pos - self.p2_pos

        # Combine escape directions
        direction = vec1 / (np.linalg.norm(vec1) + 1e-6) + \
                    vec2 / (np.linalg.norm(vec2) + 1e-6)

        # norm = np.linalg.norm(direction) + 1e-6
        # direction = direction / (np.linalg.norm(direction) + 1e-6)

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
        # self.p1_pos = np.clip(self.p1_pos, -self.world_limit, self.world_limit)
        # self.p2_pos = np.clip(self.p2_pos, -self.world_limit, self.world_limit)
        # self.evader_pos = np.clip(self.evader_pos, -self.world_limit, self.world_limit)

        self.step_count += 1

        rewards, done = self.compute_rewards()

        return self.get_states(), rewards, done

    def compute_rewards(self):

        d1 = np.linalg.norm(self.p1_pos - self.evader_pos)
        d2 = np.linalg.norm(self.p2_pos - self.evader_pos)

        # Distance shaping (small)
        r = -0.1 * (d1 + d2)

        done = False

        # Capture condition
        if d1 < self.capture_radius or d2 < self.capture_radius:
            r += 200
            done = True

        # Episode timeout
        if self.step_count >= self.max_steps:
            done = True

        return (r, r), done


    def get_states(self):
        return {
            "p1": self.p1_pos.copy(),
            "p2": self.p2_pos.copy(),
            "evader": self.evader_pos.copy()
        }

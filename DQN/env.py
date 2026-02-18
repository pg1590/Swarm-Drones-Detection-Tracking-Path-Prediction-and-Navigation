import numpy as np

class PursuitEnv:
    def __init__(self):
        self.dt = 1
        self.max_steps = 200
        self.capture_radius = 0.6

        self.max_speed_pursuer = 1.0
        self.max_speed_evader=0.7
        self.max_acc = 0.5
        self.world_limit = 5.0

        self.reset()

    def reset(self):
        self.p1_pos = np.array([-2.0, 0.0, 0.0])
        self.p2_pos = np.array([2.0, 0.0, 0.0])
        self.evader_pos = np.array([0.0, 2.0, 0.0])


        self.p1_vel = np.zeros(3)
        self.p2_vel = np.zeros(3)
        self.evader_vel = np.zeros(3)

        self.step_count = 0
        return self.get_states()

    def step(self, a1, a2, a_e):
        a1 = np.clip(a1, -self.max_acc, self.max_acc)
        a2 = np.clip(a2, -self.max_acc, self.max_acc)
        a_e = np.clip(a_e, -self.max_acc, self.max_acc)

        self.p1_vel = a1
        self.p2_vel = a2
        self.evader_vel += 0.5 * a_e * self.dt


        self.p1_vel = self.clip_speed(self.p1_vel, self.max_speed_pursuer)
        self.p2_vel = self.clip_speed(self.p2_vel, self.max_speed_pursuer)
        self.evader_vel = self.clip_speed(self.evader_vel, self.max_speed_evader)


        self.p1_pos += self.p1_vel * self.dt
        self.p2_pos += self.p2_vel * self.dt
        self.evader_pos += self.evader_vel * self.dt

        self.step_count += 1

        rewards, done = self.compute_rewards()
        return self.get_states(), rewards, done

    def clip_speed(self, v, max_speed):
        norm = np.linalg.norm(v)
        if norm > max_speed:
            return v / norm * max_speed
        return v


    def get_states(self):
        return {
            "p1": self.p1_pos.copy(),
            "p2": self.p2_pos.copy(),
            "evader": self.evader_pos.copy()
        }

    def compute_rewards(self):

        # --- Distances ---
        d1 = np.linalg.norm(self.p1_pos - self.evader_pos)
        d2 = np.linalg.norm(self.p2_pos - self.evader_pos)

        # --- Base distance reward ---
        r1 = -d1*0.1
        r2 = -d2*0.1

        # --- Cooperative alignment bonus ---
        v1 = self.p1_pos - self.evader_pos
        v2 = self.p2_pos - self.evader_pos

        # --- Capture condition ---
        done = False

        if d1 < self.capture_radius or d2 < self.capture_radius:
            r1 += 200
            r2 += 200
            done = True

        if self.step_count >= self.max_steps:
            done = True

        return (r1, r2), done

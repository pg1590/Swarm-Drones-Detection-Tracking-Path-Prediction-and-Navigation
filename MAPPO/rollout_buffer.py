import numpy as np


class RolloutBuffer:

    def __init__(self):

        self.clear()

    def clear(self):

        self.obs = []

        self.global_states = []

        self.actions = []

        self.log_probs = []

        self.rewards = []

        self.dones = []

        self.values = []

    def store(
        self,
        obs,
        global_state,
        action,
        log_prob,
        reward,
        done,
        value
    ):

        self.obs.append(obs)

        self.global_states.append(global_state)

        self.actions.append(action)

        self.log_probs.append(log_prob)

        self.rewards.append(reward)

        self.dones.append(done)

        self.values.append(value)

    def get(self):

        return {

            "obs": np.array(self.obs),

            "global_states": np.array(
                self.global_states
            ),

            "actions": np.array(self.actions),

            "log_probs": np.array(
                self.log_probs
            ),

            "rewards": np.array(
                self.rewards
            ),

            "dones": np.array(self.dones),

            "values": np.array(self.values)
        }
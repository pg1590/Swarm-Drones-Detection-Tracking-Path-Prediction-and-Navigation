import torch
import numpy as np


class RolloutBuffer:

    def __init__(self):

        self.clear()

    def clear(self):

        # =====================================================
        # OBSERVATIONS
        # =====================================================

        self.obs = []
        self.states = []

        # =====================================================
        # ACTIONS
        # =====================================================

        self.actions = []

        # =====================================================
        # PPO STUFF
        # =====================================================

        self.log_probs = []
        self.values = []

        # =====================================================
        # REWARDS / DONE
        # =====================================================

        self.rewards = []
        self.dones = []

        # =====================================================
        # RECURRENT HIDDEN STATES
        # =====================================================

        self.actor_hidden_states = []
        self.critic_hidden_states = []

    def add(
        self,
        obs,
        state,
        action,
        log_prob,
        value,
        reward,
        done,
        actor_hidden,
        critic_hidden
    ):

        self.obs.append(obs)
        self.states.append(state)

        self.actions.append(action)

        self.log_probs.append(log_prob)
        self.values.append(value)

        self.rewards.append(reward)
        self.dones.append(done)

        self.actor_hidden_states.append(actor_hidden)
        self.critic_hidden_states.append(critic_hidden)

    def compute_returns_and_advantages(
        self,
        last_value,
        gamma=0.99,
        lam=0.95
    ):

        advantages = []

        gae = 0

        values = self.values + [last_value]

        for step in reversed(range(len(self.rewards))):

            delta = (
                self.rewards[step]
                + gamma * values[step + 1] * (1 - self.dones[step])
                - values[step]
            )

            gae = (
                delta
                + gamma * lam * (1 - self.dones[step]) * gae
            )

            advantages.insert(0, gae)

        returns = [
            adv + val
            for adv, val in zip(advantages, self.values)
        ]

        self.advantages = advantages
        self.returns = returns

    def get_tensors(self, device):

        obs = torch.tensor(
            np.array(self.obs),
            dtype=torch.float32
        ).to(device)

        states = torch.tensor(
            np.array(self.states),
            dtype=torch.float32
        ).to(device)

        actions = torch.tensor(
            np.array(self.actions),
            dtype=torch.float32
        ).to(device)

        log_probs = torch.tensor(
            np.array(self.log_probs),
            dtype=torch.float32
        ).to(device)

        values = torch.tensor(
            np.array(self.values),
            dtype=torch.float32
        ).to(device)

        returns = torch.tensor(
            np.array(self.returns),
            dtype=torch.float32
        ).to(device)

        advantages = torch.tensor(
            np.array(self.advantages),
            dtype=torch.float32
        ).to(device)

        dones = torch.tensor(
            np.array(self.dones),
            dtype=torch.float32
        ).to(device)

        actor_hidden_states = torch.cat(
            self.actor_hidden_states,
            dim=1
        ).to(device)

        critic_hidden_states = torch.cat(
            self.critic_hidden_states,
            dim=1
        ).to(device)

        return (
            obs,
            states,
            actions,
            log_probs,
            values,
            returns,
            advantages,
            dones,
            actor_hidden_states,
            critic_hidden_states
        )
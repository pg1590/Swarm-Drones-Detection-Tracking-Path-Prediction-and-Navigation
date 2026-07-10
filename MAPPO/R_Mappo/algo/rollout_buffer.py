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

        # Terminated: true MDP terminal (no bootstrap).
        # Truncated: episode cut short; bootstrap_values holds
        # V(s_{t+1}) computed at rollout time, before the reset.
        self.terminateds = []
        self.truncateds = []
        self.bootstrap_values = []

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
        terminated,
        truncated,
        actor_hidden,
        critic_hidden,
        bootstrap_value=0.0
    ):

        self.obs.append(obs)
        self.states.append(state)

        self.actions.append(action)

        self.log_probs.append(log_prob)
        self.values.append(value)

        self.rewards.append(reward)
        self.dones.append(terminated or truncated)

        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        self.bootstrap_values.append(bootstrap_value)

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

            if self.terminateds[step]:
                next_value = 0.0
            elif self.truncateds[step]:
                # Partial-episode bootstrapping (Pardo et al. 2018):
                # V(s_{t+1}) recorded at rollout time, before the reset.
                next_value = self.bootstrap_values[step]
            else:
                next_value = values[step + 1]

            delta = (
                self.rewards[step]
                + gamma * next_value
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

        actor_hidden_states = torch.stack(
            self.actor_hidden_states,
            dim=1
        ).squeeze(2).to(device)

        critic_hidden_states = torch.stack(
            self.critic_hidden_states,
            dim=1
        ).squeeze(2).to(device)

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

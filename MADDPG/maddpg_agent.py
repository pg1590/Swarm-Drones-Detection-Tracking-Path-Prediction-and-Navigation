import torch
import torch.nn.functional as F
import numpy as np
import copy

from actor import Actor
from critic import Critic

class MADDPG:
    def __init__(self, state_dim, action_dim):

        # -----------------------------
        # Independent actors
        # -----------------------------
        self.actor1 = Actor(state_dim, action_dim)
        self.actor2 = Actor(state_dim, action_dim)

        self.actor1_target = copy.deepcopy(self.actor1)
        self.actor2_target = copy.deepcopy(self.actor2)

        # -----------------------------
        # Shared critic
        # -----------------------------
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)

        # -----------------------------
        # Optimizers
        # -----------------------------
        self.actor1_optimizer = torch.optim.Adam(
            self.actor1.parameters(), lr=1e-4
        )

        self.actor2_optimizer = torch.optim.Adam(
            self.actor2.parameters(), lr=1e-4
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=1e-3
        )

        self.gamma = 0.95
        self.tau = 0.01

    def select_action(self, state, agent_id, noise_std=0.1):

        state = torch.FloatTensor(state).unsqueeze(0)

        if agent_id == 1:
            action = self.actor1(state).detach().numpy()[0]
        else:
            action = self.actor2(state).detach().numpy()[0]

        action += noise_std * np.random.randn(len(action))

        return np.clip(action, -1.0, 1.0)

    def update(self, replay_buffer, batch_size):

        s1, s2, a1, a2, r, s1_next, s2_next, done = replay_buffer.sample(batch_size)

        s1 = torch.FloatTensor(s1)
        s2 = torch.FloatTensor(s2)
        a1 = torch.FloatTensor(a1)
        a2 = torch.FloatTensor(a2)
        r = torch.FloatTensor(r).unsqueeze(1)
        s1_next = torch.FloatTensor(s1_next)
        s2_next = torch.FloatTensor(s2_next)
        done = torch.FloatTensor(done).unsqueeze(1)

        # --- Critic update ---
        with torch.no_grad():
            a1_next = self.actor1_target(s1_next)
            a2_next = self.actor2_target(s2_next)

            target_Q = self.critic_target(s1_next, s2_next, a1_next, a2_next)
            y = r + self.gamma * target_Q * (1 - done)

        current_Q = self.critic(s1, s2, a1, a2)
        critic_loss = F.mse_loss(current_Q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #    -----------------------------
        # Actor 1 update
        # -----------------------------
        a1_pred = self.actor1(s1)
        a2_fixed = self.actor2(s2).detach()

        actor1_loss = -self.critic(
            s1, s2,
            a1_pred,
            a2_fixed
        ).mean()

        self.actor1_optimizer.zero_grad()
        actor1_loss.backward()
        self.actor1_optimizer.step()

        # -----------------------------
        # Actor 2 update
        # -----------------------------
        a1_fixed = self.actor1(s1).detach()
        a2_pred = self.actor2(s2)

        actor2_loss = -self.critic(
            s1, s2,
            a1_fixed,
            a2_pred
        ).mean()

        self.actor2_optimizer.zero_grad()
        actor2_loss.backward()
        self.actor2_optimizer.step()

        # -----------------------------
        # Actor 1 target update
        # -----------------------------
        for target_param, param in zip(
            self.actor1_target.parameters(),
            self.actor1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data +
                (1 - self.tau) * target_param.data
            )

        # -----------------------------
        # Actor 2 target update
        # -----------------------------
        for target_param, param in zip(
            self.actor2_target.parameters(),
            self.actor2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data +
                (1 - self.tau) * target_param.data
            )

        # -----------------------------
        # Critic target update
        # -----------------------------
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data +
                (1 - self.tau) * target_param.data
            )
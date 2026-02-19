import torch
import torch.nn.functional as F
import numpy as np
import copy

from actor import Actor
from critic import Critic

class MADDPG:
    def __init__(self, state_dim, action_dim):

        self.actor = Actor(state_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = 0.95
        self.tau = 0.01

    def select_action(self, state, noise_std=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
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
            a1_next = self.actor_target(s1_next)
            a2_next = self.actor_target(s2_next)

            target_Q = self.critic_target(s1_next, s2_next, a1_next, a2_next)
            y = r + self.gamma * target_Q * (1 - done)

        current_Q = self.critic(s1, s2, a1, a2)
        critic_loss = F.mse_loss(current_Q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        a1_pred = self.actor(s1)
        a2_pred = self.actor(s2)

        actor_loss = -self.critic(s1, s2, a1_pred, a2_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft update ---
        for target_param, param in zip(self.actor_target.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Mean action output
        self.mean_layer = nn.Linear(128, action_dim)

        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        """
        Returns action distribution
        """

        x = self.network(obs)

        mean = self.mean_layer(x)

        # Clamp mean to safe range
        mean = torch.tanh(mean)

        std = torch.exp(self.log_std)

        dist = Normal(mean, std)

        return dist

    def get_action(self, obs):
        """
        Sample action for training
        """

        dist = self.forward(obs)

        action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate(self, obs, action):
        """
        Used during PPO update
        """

        dist = self.forward(obs)

        log_prob = dist.log_prob(action).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy
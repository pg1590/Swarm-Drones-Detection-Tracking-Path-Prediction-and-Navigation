import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, global_state_dim):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(global_state_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, global_state):

        value = self.network(global_state)

        return value
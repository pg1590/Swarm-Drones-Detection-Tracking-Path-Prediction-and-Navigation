import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        input_dim = 2 * state_dim + 2 * action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, s1, s2, a1, a2):
        x = torch.cat([s1, s2, a1, a2], dim=1)
        return self.net(x)

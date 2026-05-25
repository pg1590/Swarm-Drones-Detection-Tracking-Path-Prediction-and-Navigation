import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# RECURRENT ACTOR
# ============================================================

class RecurrentActor(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=128
    ):

        super(RecurrentActor, self).__init__()

        self.hidden_dim = hidden_dim

        # Feature extractor
        self.fc1 = nn.Linear(obs_dim, hidden_dim)

        # GRU memory
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Action mean
        self.fc_mean = nn.Linear(hidden_dim, action_dim)

        # Learnable std
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * -0.5
        )

    def forward(self, obs, hidden_state):

        """
        obs shape:
            (batch, seq_len, obs_dim)

        hidden_state shape:
            (1, batch, hidden_dim)
        """

        x = F.relu(self.fc1(obs))

        x, next_hidden = self.gru(x, hidden_state)

        mean = self.fc_mean(x)

        std = torch.exp(self.log_std)

        dist = Normal(mean, std)

        return dist, next_hidden

    def init_hidden(self, batch_size=1):

        return torch.zeros(
            1,
            batch_size,
            self.hidden_dim
        ).to(device)


# ============================================================
# RECURRENT CRITIC
# ============================================================

class RecurrentCritic(nn.Module):

    def __init__(
        self,
        state_dim,
        hidden_dim=128
    ):

        super(RecurrentCritic, self).__init__()

        self.hidden_dim = hidden_dim

        # State encoder
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # GRU memory
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Value output
        self.fc_value = nn.Linear(hidden_dim, 1)

    def forward(self, state, hidden_state):

        """
        state shape:
            (batch, seq_len, state_dim)

        hidden_state shape:
            (1, batch, hidden_dim)
        """

        x = F.relu(self.fc1(state))

        x, next_hidden = self.gru(x, hidden_state)

        value = self.fc_value(x)

        return value, next_hidden

    def init_hidden(self, batch_size=1):

        return torch.zeros(
            1,
            batch_size,
            self.hidden_dim
        ).to(device)
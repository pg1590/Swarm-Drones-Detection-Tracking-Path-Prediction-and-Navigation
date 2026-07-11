import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CANONICAL ORTHOGONAL INITIALIZATION
# ============================================================

def init_layer(layer, gain=1.0):

    """
    Official MAPPO init (on-policy, utils/util.py::init):
    orthogonal weights with the given gain, zero bias.
    """

    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)

    return layer


def init_gru(gru):

    """
    Official MAPPO GRU init (on-policy, utils/rnn.py::RNNLayer):
    orthogonal weight matrices, zero biases.
    """

    for name, param in gru.named_parameters():

        if "bias" in name:
            nn.init.constant_(param, 0.0)
        elif "weight" in name:
            nn.init.orthogonal_(param)

    return gru


# ============================================================
# MASK-AWARE GRU FORWARD
# ============================================================

def masked_gru_forward(gru, x, hidden_state, masks):

    """
    Runs a GRU over a sequence while zeroing the hidden state at
    episode boundaries inside the sequence.

    x:            (batch, seq_len, features)
    hidden_state: (1, batch, hidden_dim)
    masks:        (batch, seq_len) with 1.0 = carry hidden into this
                  step, 0.0 = this step starts a new episode.

    Segments between boundary steps use a single fused GRU call.
    """

    seq_len = x.size(1)

    boundaries = (
        (masks[:, 1:] == 0.0)
        .any(dim=0)
        .nonzero()
        .squeeze(-1)
        .add(1)
        .tolist()
    )

    boundaries = [0] + boundaries + [seq_len]

    outputs = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):

        hidden_state = hidden_state * masks[:, start].view(1, -1, 1)

        out, hidden_state = gru(
            x[:, start:end],
            hidden_state
        )

        outputs.append(out)

    return torch.cat(outputs, dim=1), hidden_state


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
        self.fc1 = init_layer(
            nn.Linear(obs_dim, hidden_dim),
            gain=nn.init.calculate_gain("relu")
        )

        self.norm1 = nn.LayerNorm(hidden_dim)

        # GRU memory
        self.gru = init_gru(
            nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )
        )

        # Action mean: small output gain so the initial policy is
        # near-centered (official MAPPO ACTLayer, gain=0.01).
        self.fc_mean = init_layer(
            nn.Linear(hidden_dim, action_dim),
            gain=0.01
        )

        # Learnable std
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * -0.5
        )

    def forward(self, obs, hidden_state, masks=None):

        """
        obs shape:
            (batch, seq_len, obs_dim)

        hidden_state shape:
            (1, batch, hidden_dim)

        masks shape (optional, training only):
            (batch, seq_len), 0.0 where a new episode starts.
        """

        x = F.relu(
            self.norm1(
                self.fc1(obs)
            )
        )

        if masks is None:
            x, next_hidden = self.gru(x, hidden_state)
        else:
            x, next_hidden = masked_gru_forward(
                self.gru,
                x,
                hidden_state,
                masks
            )

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
        self.fc1 = init_layer(
            nn.Linear(state_dim, hidden_dim),
            gain=nn.init.calculate_gain("relu")
        )

        self.norm1 = nn.LayerNorm(hidden_dim)

        # GRU memory
        self.gru = init_gru(
            nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )
        )

        # Value output (official MAPPO r_critic.py, default gain 1)
        self.fc_value = init_layer(
            nn.Linear(hidden_dim, 1),
            gain=1.0
        )

    def forward(self, state, hidden_state, masks=None):

        """
        state shape:
            (batch, seq_len, state_dim)

        hidden_state shape:
            (1, batch, hidden_dim)

        masks shape (optional, training only):
            (batch, seq_len), 0.0 where a new episode starts.
        """

        x = F.relu(
            self.norm1(
                self.fc1(state)
            )
        )

        if masks is None:
            x, next_hidden = self.gru(x, hidden_state)
        else:
            x, next_hidden = masked_gru_forward(
                self.gru,
                x,
                hidden_state,
                masks
            )

        value = self.fc_value(x)

        return value, next_hidden

    def init_hidden(self, batch_size=1):

        return torch.zeros(
            1,
            batch_size,
            self.hidden_dim
        ).to(device)

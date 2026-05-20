import numpy as np
import torch


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SET RANDOM SEED
# ============================================================

def set_seed(seed):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


# ============================================================
# CONVERT TO TENSOR
# ============================================================

def to_tensor(x):

    return torch.FloatTensor(x).to(device)


# ============================================================
# RESET GRU HIDDEN STATES
# ============================================================

def reset_hidden_state(hidden_state):

    hidden_state.zero_()

    return hidden_state


# ============================================================
# DETACH HIDDEN STATES
# ============================================================

def detach_hidden_state(hidden_state):

    return hidden_state.detach()


# ============================================================
# LINEAR LR DECAY
# ============================================================

def linear_lr_decay(
    optimizer,
    current_step,
    total_steps,
    initial_lr
):

    lr = initial_lr * (
        1 - (current_step / float(total_steps))
    )

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr


# ============================================================
# NORMALIZE ARRAY
# ============================================================

def normalize(x):

    x = np.array(x)

    return (
        (x - x.mean())
        / (x.std() + 1e-8)
    )


# ============================================================
# COMPUTE DISTANCE
# ============================================================

def euclidean_distance(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.linalg.norm(a - b)


# ============================================================
# CLIP ACTIONS
# ============================================================

def clip_actions(actions, low=-1.0, high=1.0):

    return np.clip(actions, low, high)


# ============================================================
# PRINT CUDA INFO
# ============================================================

def print_cuda_info():

    print("===================================")

    print("CUDA AVAILABLE:", torch.cuda.is_available())

    if torch.cuda.is_available():

        print("GPU:", torch.cuda.get_device_name(0))

        print(
            "CUDA VERSION:",
            torch.version.cuda
        )

    print("DEVICE:", device)

    print("===================================")
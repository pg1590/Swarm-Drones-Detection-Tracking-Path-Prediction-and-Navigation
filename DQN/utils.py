import numpy as np
import torch
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def create_action_space(eta):
    actions = []
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]


    for ax in values:
        for ay in values:
            for az in values:
                actions.append(np.array([ax, ay, az]))

    return actions


def index_to_action(index, action_list):
    return action_list[index]

def compute_distance(a, b):
    return np.linalg.norm(a - b)

def build_joint_state(my_pos, my_vel, other_pos, other_vel, evader_pos):
    scale_pos = 5.0
    scale_vel = 1.0

    return np.concatenate([
        my_pos / scale_pos,
        my_vel / scale_vel,
        other_pos / scale_pos,
        other_vel / scale_vel,
        evader_pos / scale_pos
    ])



def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1 - tau) * target_param.data
        )

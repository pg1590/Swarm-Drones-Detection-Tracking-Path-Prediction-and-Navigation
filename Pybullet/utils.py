import numpy as np


def normalize_vector(vec, eps=1e-6):

    norm = np.linalg.norm(vec)

    if norm < eps:
        return vec

    return vec / norm


def compute_distance(a, b):

    return np.linalg.norm(a - b)


def compute_team_center(p1, p2):

    return (p1 + p2) / 2.0


def compute_surround_metric(
    p1,
    p2,
    target
):
    """
    Measures how well the target is trapped
    between agents.

    Larger values imply better enclosure.
    """

    d1 = compute_distance(p1, target)

    d2 = compute_distance(p2, target)

    inter_agent = compute_distance(p1, p2)

    metric = (
        inter_agent
        /
        (d1 + d2 + 1e-6)
    )

    return metric


def soft_update(source, target, tau):
    """
    Optional utility if later needed
    for hybrid algorithms.
    """

    for target_param, param in zip(
        target.parameters(),
        source.parameters()
    ):

        target_param.data.copy_(

            tau * param.data
            +
            (1 - tau) * target_param.data
        )

def enclosure_reward(p1, p2, target):

    v1 = p1 - target
    v2 = p2 - target

    v1 = v1 / (
        np.linalg.norm(v1) + 1e-6
    )

    v2 = v2 / (
        np.linalg.norm(v2) + 1e-6
    )

    dot = np.clip(
        np.dot(v1, v2),
        -1.0,
        1.0
    )

    angle = np.arccos(dot)

    # normalize
    angle /= np.pi

    return angle
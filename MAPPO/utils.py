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

import numpy as np


def compute_swarm_metrics(
    p1_pos,
    p2_pos,
    evader_pos,
    predicted_evader_pos=None,
    capture=False
):

    metrics = {}

    # -----------------------------------
    # Pursuer Separation
    # -----------------------------------

    separation = np.linalg.norm(
        p1_pos - p2_pos
    )

    metrics["separation"] = separation

    # -----------------------------------
    # Distances to Evader
    # -----------------------------------

    d1 = np.linalg.norm(
        p1_pos - evader_pos
    )

    d2 = np.linalg.norm(
        p2_pos - evader_pos
    )

    metrics["p1_evader_dist"] = d1
    metrics["p2_evader_dist"] = d2

    metrics["mean_evader_dist"] = (
        d1 + d2
    ) / 2.0

    # -----------------------------------
    # Enclosure Angle
    # -----------------------------------

    v1 = p1_pos - evader_pos
    v2 = p2_pos - evader_pos

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 > 1e-6 and norm2 > 1e-6:

        cosine = np.dot(v1, v2) / (
            norm1 * norm2
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )

        angle = np.degrees(
            np.arccos(cosine)
        )

    else:
        angle = 0.0

    metrics["enclosure_angle"] = angle

    # -----------------------------------
    # Team Center
    # -----------------------------------

    team_center = (
        p1_pos + p2_pos
    ) / 2.0

    team_center_dist = np.linalg.norm(
        team_center - evader_pos
    )

    metrics["team_center_dist"] = (
        team_center_dist
    )

    # -----------------------------------
    # Interception Error
    # -----------------------------------

    if predicted_evader_pos is not None:

        pred_error = np.linalg.norm(
            team_center -
            predicted_evader_pos
        )

        metrics[
            "interception_error"
        ] = pred_error

    # -----------------------------------
    # Capture
    # -----------------------------------

    metrics["capture"] = int(capture)

    return metrics
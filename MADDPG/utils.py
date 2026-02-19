import numpy as np

def build_joint_state(my_pos, my_vel,
                      other_pos, other_vel,
                      evader_pos):

    scale_pos = 5.0
    scale_vel = 1.0

    return np.concatenate([
        my_pos / scale_pos,
        my_vel / scale_vel,
        other_pos / scale_pos,
        other_vel / scale_vel,
        evader_pos / scale_pos
    ])

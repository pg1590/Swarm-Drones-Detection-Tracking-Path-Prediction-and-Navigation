import numpy as np

def build_joint_state(my_pos, my_vel,
                      other_pos, other_vel,
                      evader_pos, evader_vel):

    scale_pos = 10.0
    scale_vel = 2.0

    relative_target = evader_pos - my_pos
    relative_other = other_pos - my_pos

    return np.concatenate([
        my_pos / scale_pos,
        my_vel / scale_vel,

        relative_other / scale_pos,
        other_vel / scale_vel,

        relative_target / scale_pos,
        evader_vel / scale_vel
    ])
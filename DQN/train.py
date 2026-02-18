import numpy as np
import torch

from env import PursuitEnv
from agent import PursuerAgent
from buffer import ReplayBuffer
from utils import build_joint_state, create_action_space, index_to_action

# ==========================
# HYPERPARAMETERS
# ==========================

MAX_EPISODES = 3000
MAX_STEPS = 400
BATCH_SIZE = 256
GAMMA = 0.95
ALPHA = 0.5  # cooperative reward weight
ETA = 0.5    # max acceleration discretization

# ==========================
# ACTION SPACE
# ==========================

action_list = create_action_space(ETA)
action_dim = len(action_list)

# State dimension:
# my_pos(3) + my_vel(3) + other_pos(3) + other_vel(3) + evader pos(3)
state_dim = 15


# ==========================
# SCRIPTED EVADER (BASELINE)
# ==========================

def scripted_evader_policy():
    return np.random.uniform(-ETA, ETA, size=3)


# ==========================
# MAIN TRAINING
# ==========================

def main():

    env = PursuitEnv()

    shared_agent = PursuerAgent(state_dim, action_dim)


    buffer = ReplayBuffer(100000)

    success_count = 0

    for episode in range(MAX_EPISODES):

        states = env.reset()
        done = False
        episode_reward = 0

        for step in range(MAX_STEPS):

            # Extract positions
            p1_pos = states["p1"]
            p2_pos = states["p2"]
            e_pos = states["evader"]

            p1_vel = env.p1_vel
            p2_vel = env.p2_vel

            # Build joint states (mean-field for 2 agents = other agent)
            s1 = build_joint_state(p1_pos, p1_vel, p2_pos, p2_vel, e_pos)
            s2 = build_joint_state(p2_pos, p2_vel, p1_pos, p1_vel, e_pos)


            # Select actions (discrete index)
            a1_index = shared_agent.select_action(s1)
            a2_index = shared_agent.select_action(s2)


            # Convert index → acceleration vector
            a1 = index_to_action(a1_index, action_list)
            a2 = index_to_action(a2_index, action_list)

            # Evader action
            a_e = scripted_evader_policy()

            # Step environment
            next_states, (r1, r2), done = env.step(a1, a2, a_e)

            # Cooperative reward reallocation
            r1_hat = r1 + ALPHA * r2
            r2_hat = r2 + ALPHA * r1

            episode_reward += r1_hat + r2_hat

            # Build next joint states
            next_p1_pos = next_states["p1"]
            next_p2_pos = next_states["p2"]

            next_e_pos = next_states["evader"]

            next_s1 = build_joint_state(next_p1_pos, env.p1_vel,
                            next_p2_pos, env.p2_vel,
                            next_e_pos)

            next_s2 = build_joint_state(next_p2_pos, env.p2_vel,
                            next_p1_pos, env.p1_vel,
                            next_e_pos)


            # Store in replay buffer
            buffer.push(s1, a1_index, r1_hat, next_s1, done)
            buffer.push(s2, a2_index, r2_hat, next_s2, done)

            states = next_states

            if done:
                if r1 > 40 or r2 > 40:
                    success_count += 1
                break

        # ======================
        # TRAIN AFTER EPISODE
        # ======================

        if len(buffer) > BATCH_SIZE:
            shared_agent.update(buffer.sample(BATCH_SIZE))

        # Logging
        if episode % 50 == 0:
            success_rate = success_count / (episode + 1)
            print(f"Episode {episode}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Success Rate: {success_rate:.3f}")
            print("-" * 40)


    print("Training complete.")
    print("Final Success Rate:", success_count / MAX_EPISODES)


if __name__ == "__main__":
    main()

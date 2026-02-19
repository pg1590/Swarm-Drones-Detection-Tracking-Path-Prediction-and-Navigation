from env import PursuitEnv
from maddpg_agent import MADDPG
from replay_buffer import ReplayBuffer
from utils import build_joint_state

import numpy as np

MAX_EPISODES = 5000
BATCH_SIZE = 256

def main():

    env = PursuitEnv()
    state_dim = 15
    action_dim = 3

    agent = MADDPG(state_dim, action_dim)
    buffer = ReplayBuffer(100000)

    success_count = 0

    for episode in range(MAX_EPISODES):

        states = env.reset()
        done = False
        episode_reward = 0

        while not done:

            p1_pos = states["p1"]
            p2_pos = states["p2"]
            e_pos = states["evader"]

            s1 = build_joint_state(p1_pos, env.p1_vel,
                                   p2_pos, env.p2_vel,
                                   e_pos)

            s2 = build_joint_state(p2_pos, env.p2_vel,
                                   p1_pos, env.p1_vel,
                                   e_pos)

            a1 = agent.select_action(s1)
            a2 = agent.select_action(s2)

            next_states, rewards, done = env.step(a1, a2, np.zeros(3))

            r = rewards[0]  # shared reward assumption

            next_s1 = build_joint_state(next_states["p1"], env.p1_vel,
                                        next_states["p2"], env.p2_vel,
                                        next_states["evader"])

            next_s2 = build_joint_state(next_states["p2"], env.p2_vel,
                                        next_states["p1"], env.p1_vel,
                                        next_states["evader"])

            buffer.push(s1, s2, a1, a2, r, next_s1, next_s2, done)

            states = next_states
            episode_reward += r

            if len(buffer) > BATCH_SIZE:
                agent.update(buffer, BATCH_SIZE)

        if rewards[0] > 100:
            success_count += 1

        if episode % 50 == 0:
            print(f"Episode {episode}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Success Rate: {success_count/(episode+1):.3f}")
            print("-"*40)

    print("Training complete.")

if __name__ == "__main__":
    main()

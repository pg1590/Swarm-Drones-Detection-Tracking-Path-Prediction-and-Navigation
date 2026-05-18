from env import PursuitEnv
from maddpg_agent import MADDPG
from replay_buffer import ReplayBuffer
from utils import build_joint_state
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

MAX_EPISODES = 5000
BATCH_SIZE = 256

def main():
    os.makedirs("plots", exist_ok=True)
    env = PursuitEnv()
    state_dim = 18
    action_dim = 3

    agent = MADDPG(state_dim, action_dim)
    buffer = ReplayBuffer(100000)

    success_count = 0

    for episode in range(MAX_EPISODES):
        env.total_episodes = episode
        agent.noise1.reset()
        agent.noise2.reset()
        states = env.reset()
        p1_traj = []
        p2_traj = []
        capture_metric_history = []
        evader_traj = []

        separation_history = []
        angle_history = []
        done = False
        episode_reward = 0

        while not done:
            p1_traj.append(env.p1_pos.copy())
            p2_traj.append(env.p2_pos.copy())
            evader_traj.append(env.evader_pos.copy())
            p1_pos = states["p1"]
            p2_pos = states["p2"]
            e_pos = states["evader"]

            s1 = build_joint_state(
                    p1_pos, env.p1_vel,
                    p2_pos, env.p2_vel,
                    e_pos, env.evader_vel
                )
            s2 = build_joint_state(
                p2_pos, env.p2_vel,
                p1_pos, env.p1_vel,
                e_pos, env.evader_vel
            )

            a1 = agent.select_action(s1, agent_id=1)
            a2 = agent.select_action(s2, agent_id=2)
            next_states, rewards, done = env.step(a1, a2, np.zeros(3))

            r = rewards[0]  # shared reward assumption

            next_s1 = build_joint_state(next_states["p1"], env.p1_vel,
                                        next_states["p2"], env.p2_vel,
                                        next_states["evader"],env.evader_vel)

            next_s2 = build_joint_state(next_states["p2"], env.p2_vel,
                                        next_states["p1"], env.p1_vel,
                                        next_states["evader"],env.evader_vel)

            buffer.push(s1, s2, a1, a2, r, next_s1, next_s2, done)

            states = next_states
            episode_reward += r


            if len(buffer) > 5000:
                agent.update(buffer, BATCH_SIZE)

            # --------------------------------
            # Pursuer separation
            # --------------------------------
            sep = np.linalg.norm(env.p1_pos - env.p2_pos)
            separation_history.append(sep)

            # --------------------------------
            # Angular enclosure
            # --------------------------------
            v1 = env.p1_pos - env.evader_pos
            v2 = env.p2_pos - env.evader_pos

            v1 /= (np.linalg.norm(v1) + 1e-6)
            v2 /= (np.linalg.norm(v2) + 1e-6)
            d1 = np.linalg.norm(env.p1_pos - env.evader_pos)
            d2 = np.linalg.norm(env.p2_pos - env.evader_pos)
            capture_metric = (
                np.linalg.norm(np.cross(v1, v2))
                /
                ((d1 + d2)/2 + 1e-6)
            )

            capture_metric_history.append(capture_metric)
            dot = np.clip(np.dot(v1, v2), -1.0, 1.0)

            angle = np.degrees(np.arccos(dot))

            angle_history.append(angle)

        d1 = np.linalg.norm(env.p1_pos - env.evader_pos)
        d2 = np.linalg.norm(env.p2_pos - env.evader_pos)

        if d1 < env.capture_radius and d2 < env.capture_radius:
            success_count += 1

        if episode % 500 == 0:
            print(f"Episode {episode}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Success Rate: {success_count/(episode+1):.3f}")
            print("-"*40)
            # =====================================
            # TRAJECTORY VISUALIZATION
            # =====================================

            p1_traj_np = np.array(p1_traj)
            p2_traj_np = np.array(p2_traj)
            evader_traj_np = np.array(evader_traj)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # --------------------------------
            # Plot trajectories
            # --------------------------------
            ax.plot(
                p1_traj_np[:,0],
                p1_traj_np[:,1],
                p1_traj_np[:,2],
                label='Pursuer 1'
            )

            ax.plot(
                p2_traj_np[:,0],
                p2_traj_np[:,1],
                p2_traj_np[:,2],
                label='Pursuer 2'
            )

            ax.plot(
                evader_traj_np[:,0],
                evader_traj_np[:,1],
                evader_traj_np[:,2],
                label='Evader'
            )

            # --------------------------------
            # Start points
            # --------------------------------
            ax.scatter(
                p1_traj_np[0,0],
                p1_traj_np[0,1],
                p1_traj_np[0,2],
                s=80
            )

            ax.scatter(
                p2_traj_np[0,0],
                p2_traj_np[0,1],
                p2_traj_np[0,2],
                s=80
            )

            ax.scatter(
                evader_traj_np[0,0],
                evader_traj_np[0,1],
                evader_traj_np[0,2],
                s=80
            )

            # --------------------------------
            # Labels
            # --------------------------------
            ax.set_title(f"Episode {episode}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            ax.legend()

            plt.savefig(f"plots/trajectory_ep_{episode}.png")
            plt.close()
            # =====================================
            # SEPARATION + ANGLE PLOTS
            # =====================================
            fig2, axarr = plt.subplots(3, 1, figsize=(8, 10))
            # --------------------------------
            # Trap metric
            # --------------------------------
            axarr[2].plot(capture_metric_history)
            axarr[2].set_title("Trap Metric")
            axarr[2].set_ylabel("Trap Quality")
            axarr[2].set_xlabel("Timestep")
            # --------------------------------
            # Separation
            # --------------------------------
            axarr[0].plot(separation_history)
            axarr[0].set_title("Pursuer Separation")
            axarr[0].set_ylabel("Distance")

            # --------------------------------
            # Enclosure angle
            # --------------------------------
            axarr[1].plot(angle_history)
            axarr[1].set_title("Enclosure Angle")
            axarr[1].set_ylabel("Degrees")
            axarr[1].set_xlabel("Timestep")

            plt.tight_layout()
            plt.savefig(f"plots/metrics_ep_{episode}.png")
            plt.close()

    print("Training complete.")

if __name__ == "__main__":
    main()

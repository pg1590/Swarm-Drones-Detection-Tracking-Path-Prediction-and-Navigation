import time
import numpy as np
import torch

from envs.env import DroneSwarmEnv

from algo.r_mappo import R_MAPPO

from algo.networks import device

from algo.utils import clip_actions

from plots.trajectory_plotter import (
    plot_trajectories
)

from configs.config import *


# ============================================================
# EVALUATION CONFIG
# ============================================================

EVAL_EPISODES = 5

MODEL_TO_LOAD = BEST_MODEL_PATH

GUI = True


# ============================================================
# MAIN
# ============================================================

def main():

    print("===================================")
    print("EVALUATION MODE")
    print("DEVICE:", device)
    print("===================================")

    # ========================================================
    # ENVIRONMENT
    # ========================================================

    env = DroneSwarmEnv(
        num_drones=NUM_DRONES,
        gui=GUI,
        max_steps=MAX_STEPS
    )

    obs_dim = env.observation_space.shape[0]

    action_dim = env.action_space.shape[0]

    state_dim = (
        NUM_DRONES * 4
        + 2
    )

    # ========================================================
    # AGENT
    # ========================================================

    agent = R_MAPPO(
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    agent.load(MODEL_TO_LOAD)

    print(f"Loaded model: {MODEL_TO_LOAD}")

    # ========================================================
    # EVALUATION LOOP
    # ========================================================
    capture_threshold = 1.0

    successful_captures = 0

    for episode in range(EVAL_EPISODES):

        obs = env.reset()

        state = env.get_global_state()

        episode_reward = 0

        # ====================================================
        # GRU HIDDEN STATES
        # ====================================================

        actor_hiddens = [
            agent.actor.init_hidden(batch_size=1)
            for _ in range(NUM_DRONES)
        ]

        critic_hiddens = [
            agent.critic.init_hidden(batch_size=1)
            for _ in range(NUM_DRONES)
        ]
        # ====================================================
        # EPISODE LOOP
        # ====================================================

        for step in range(MAX_STEPS):

            actions = []

            # ================================================
            # ACTION SELECTION
            # ================================================

            with torch.no_grad():

                next_actor_hiddens = []

                next_critic_hiddens = []

                for drone_idx in range(NUM_DRONES):

                    action, _, _, \
                    next_actor_hidden, \
                    next_critic_hidden = agent.select_action(
                        obs[drone_idx],
                        state,
                        actor_hiddens[drone_idx],
                        critic_hiddens[drone_idx]
                    )

                    action = clip_actions(action)

                    actions.append(action)

                    next_actor_hiddens.append(
                        next_actor_hidden
                    )

                    next_critic_hiddens.append(
                        next_critic_hidden
                    )

            # ================================================
            # ENV STEP
            # ================================================

            next_obs, rewards, dones, _ = env.step(
                actions
            )

            next_state = env.get_global_state()

            reward = np.mean(rewards)

            # ====================================================
            # INTERCEPTION METRICS
            # ====================================================

            distances_to_target = []

            for drone_id in env.drones:

                pos, _ = p.getBasePositionAndOrientation(
                    drone_id
                )

                pos = np.array(pos[:2])

                dist = np.linalg.norm(
                    pos - env.target_pos
                )

                distances_to_target.append(dist)

            min_distance = np.min(distances_to_target)

            if min_distance < capture_threshold:

                successful_captures += 1

            done = any(dones)

            episode_reward += reward

            # ====================================================
            # ESCAPE CORRIDOR METRIC
            # ====================================================

            angles = []

            for drone_id in env.drones:

                pos, _ = p.getBasePositionAndOrientation(
                    drone_id
                )

                pos = np.array(pos[:2])

                vec = pos - env.target_pos

                angle = np.arctan2(
                    vec[1],
                    vec[0]
                )

                angles.append(angle)

            angles = np.sort(angles)

            largest_gap = 0

            for k in range(len(angles) - 1):

                gap = angles[k + 1] - angles[k]

                largest_gap = max(
                    largest_gap,
                    gap
                )

            wrap_gap = (
                2 * np.pi
                - angles[-1]
                + angles[0]
            )

            largest_gap = max(
                largest_gap,
                wrap_gap
            )

            escape_corridor = largest_gap

            # ================================================
            # UPDATE STATES
            # ================================================

            obs = next_obs

            state = next_state

            actor_hidden = next_actor_hiddens

            critic_hidden = next_critic_hiddens

            # ================================================
            # SLOW DOWN FOR VISUALIZATION
            # ================================================

            time.sleep(1 / 60)

            # ================================================
            # DONE
            # ================================================

            if done:
                break

        # ====================================================
        # EPISODE RESULTS
        # ====================================================

        print(
            f"Evaluation Episode {episode} | "
            f"Reward: {episode_reward:.2f}"
            f"Captures: {successful_captures} | "
            f"Escape Corridor: {escape_corridor:.2f}"
        )

        # ====================================================
        # SAVE TRAJECTORY PLOT
        # ====================================================

        plot_trajectories(
            env.trajectory_history,
            target_positions=env.target_history,
            save_path=f"plots/eval/eval_episode_{episode}.png",
            show_plot=False
        )

    # ========================================================
    # CLOSE
    # ========================================================

    env.close()

    print("Evaluation completed.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
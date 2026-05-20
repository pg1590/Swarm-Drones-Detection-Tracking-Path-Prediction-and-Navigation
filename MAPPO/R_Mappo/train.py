import os
import numpy as np
import torch

from envs.env import DroneSwarmEnv

from algo.r_mappo import R_MAPPO
from algo.rollout_buffer import RolloutBuffer

from algo.networks import device

from algo.utils import (
    set_seed,
    print_cuda_info,
    clip_actions
)

from plots.trajectory_plotter import (
    plot_trajectories,
    plot_rewards,
    plot_smoothed_rewards
)

from configs.config import *
# ============================================================
# CONFIG
# ============================================================



MODEL_PATH = "models/recurrent_mappo_latest.pth"

BEST_MODEL_PATH = "models/recurrent_mappo_best.pth"


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CUDA INFO
    # ========================================================

    print_cuda_info()

    # ========================================================
    # SEED
    # ========================================================

    set_seed(SEED)

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
    # BUFFER
    # ========================================================

    buffer = RolloutBuffer()

    # ========================================================
    # LOGGING
    # ========================================================

    episode_rewards = []

    best_reward = -1e9

    timestep = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for episode in range(MAX_EPISODES):

        obs = env.reset()

        state = env.get_global_state()

        episode_reward = 0

        # ====================================================
        # GRU HIDDEN STATES
        # ====================================================

        actor_hidden = agent.actor.init_hidden(
            batch_size=1
        )

        critic_hidden = agent.critic.init_hidden(
            batch_size=1
        )

        # ====================================================
        # EPISODE LOOP
        # ====================================================

        for step in range(MAX_STEPS):

            timestep += 1

            actions = []

            log_probs = []

            values = []

            # ================================================
            # ACTION SELECTION
            # ================================================

            for drone_idx in range(NUM_DRONES):

                action, log_prob, value, \
                next_actor_hidden, \
                next_critic_hidden = agent.select_action(
                    obs[drone_idx],
                    actor_hidden,
                    critic_hidden
                )

                action = clip_actions(action)

                actions.append(action)

                log_probs.append(log_prob)

                values.append(value)

            # ================================================
            # ENV STEP
            # ================================================

            next_obs, rewards, dones, _ = env.step(
                actions
            )

            next_state = env.get_global_state()

            reward = np.mean(rewards)

            done = any(dones)

            episode_reward += reward

            # ================================================
            # STORE IN BUFFER
            # ================================================

            buffer.add(
                obs=np.mean(obs, axis=0),
                state=state,
                action=np.mean(actions, axis=0),
                log_prob=np.mean(log_probs),
                value=np.mean(values),
                reward=reward,
                done=done,
                actor_hidden=actor_hidden,
                critic_hidden=critic_hidden
            )

            # ================================================
            # UPDATE STATES
            # ================================================

            obs = next_obs

            state = next_state

            actor_hidden = next_actor_hidden

            critic_hidden = next_critic_hidden

            # ================================================
            # PPO UPDATE
            # ================================================

            if timestep % UPDATE_TIMESTEPS == 0:

                with torch.no_grad():

                    state_tensor = torch.FloatTensor(
                        state
                    ).unsqueeze(0).unsqueeze(0).to(device)

                    last_value, _ = agent.critic(
                        state_tensor,
                        critic_hidden
                    )

                    last_value = last_value.item()

                buffer.compute_returns_and_advantages(
                    last_value
                )

                agent.update(buffer)

                buffer.clear()

                print(
                    f"UPDATE COMPLETED @ timestep {timestep}"
                )

            # ================================================
            # DONE
            # ================================================

            if done:
                break

        # ====================================================
        # LOGGING
        # ====================================================

        episode_rewards.append(episode_reward)

        print(
            f"Episode: {episode} | "
            f"Reward: {episode_reward:.2f}"
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if episode_reward > best_reward:

            best_reward = episode_reward

            agent.save(BEST_MODEL_PATH)

            print("Best model saved.")

        # ====================================================
        # PERIODIC SAVE
        # ====================================================

        if episode % SAVE_INTERVAL == 0:

            agent.save(MODEL_PATH)

            print("Checkpoint saved.")

            # ================================================
            # SAVE TRAJECTORY PLOT
            # ================================================

            plot_trajectories(
                env.trajectory_history,
                target_positions=None,
                save_path=f"plots/trajectories/episode_{episode}.png",
                show_plot=False
            )

            # ================================================
            # SAVE REWARD PLOTS
            # ================================================

            plot_rewards(
                episode_rewards,
                save_path="plots/rewards/reward_curve.png",
                show_plot=False
            )

            plot_smoothed_rewards(
                episode_rewards,
                save_path="plots/rewards/smoothed_reward_curve.png",
                show_plot=False
            )

    # ========================================================
    # FINAL SAVE
    # ========================================================

    agent.save(MODEL_PATH)

    env.close()

    print("Training completed.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
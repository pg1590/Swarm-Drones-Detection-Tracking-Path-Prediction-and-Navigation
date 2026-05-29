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
    plot_smoothed_rewards,
    plot_capture_rate,
    plot_mean_distance,
    plot_coverage,
    plot_escape_gap,
    plot_velocity_diversity
)

from configs.config import *


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
        max_steps=MAX_STEPS,
        control_dt=CONTROL_DT,
        max_speed=MAX_SPEED,
        velocity_response=VELOCITY_RESPONSE,
        capture_radius=CAPTURE_RADIUS,
        safe_spacing=SAFE_SPACING
    )

    obs_dim = env.observation_space.shape[0]

    action_dim = env.action_space.shape[0]

    state_dim = env.get_global_state().shape[0]

    # ========================================================
    # AGENT
    # ========================================================

    agent = R_MAPPO(
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        gamma=GAMMA,
        lam=LAMBDA,
        clip_eps=CLIP_EPS,
        epochs=PPO_EPOCHS,
        entropy_coef=ENTROPY_COEF,
        value_coef=VALUE_COEF,
        hidden_dim=HIDDEN_DIM,
        sequence_length=SEQUENCE_LENGTH
    )

    # ========================================================
    # BUFFER
    # ========================================================

    buffers = [
        RolloutBuffer()
        for _ in range(NUM_DRONES)
    ]

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

            timestep += 1

            actions = []

            log_probs = []

            values = []

            # ================================================
            # ACTION SELECTION
            # ================================================

            next_actor_hiddens = []

            next_critic_hiddens = []

            for drone_idx in range(NUM_DRONES):

                action, log_prob, value, \
                next_actor_hidden, \
                next_critic_hidden = agent.select_action(
                    obs[drone_idx],
                    state,
                    actor_hiddens[drone_idx],
                    critic_hiddens[drone_idx]
                )

                action = clip_actions(action)

                actions.append(action)

                log_probs.append(log_prob)

                values.append(value)

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

            done = any(dones)

            episode_reward += reward

            # ================================================
            # STORE IN BUFFER
            # ================================================

            for drone_idx in range(NUM_DRONES):

                buffers[drone_idx].add(
                    obs=obs[drone_idx],
                    state=state,
                    action=actions[drone_idx],
                    log_prob=log_probs[drone_idx],
                    value=values[drone_idx],
                    reward=rewards[drone_idx],
                    done=done,
                    actor_hidden=actor_hiddens[drone_idx],
                    critic_hidden=critic_hiddens[drone_idx]
                )

            # ================================================
            # UPDATE STATES
            # ================================================

            obs = next_obs

            state = next_state

            actor_hiddens = next_actor_hiddens

            critic_hiddens = next_critic_hiddens

            # ================================================
            # PPO UPDATE
            # ================================================

            if timestep % UPDATE_TIMESTEPS == 0:

                eligible_buffers = []

                for drone_idx in range(NUM_DRONES):

                    with torch.no_grad():

                        state_tensor = torch.FloatTensor(
                            state
                        ).unsqueeze(0).unsqueeze(0).to(device)

                        last_value, _ = agent.critic(
                            state_tensor,
                            critic_hiddens[drone_idx]
                        )

                        last_value = last_value.item()

                    buffers[drone_idx].compute_returns_and_advantages(
                        last_value,
                        gamma=GAMMA,
                        lam=LAMBDA
                    )

                    if len(buffers[drone_idx].obs) >= SEQUENCE_LENGTH:

                        eligible_buffers.append(
                            buffers[drone_idx]
                        )

                if len(eligible_buffers) > 0:

                    agent.update(
                        eligible_buffers
                    )

                for drone_idx in range(NUM_DRONES):
                    buffers[drone_idx].clear()

                print(
                    f"UPDATE COMPLETED @ timestep {timestep}"
                )
            # ================================================
            # DONE
            # ================================================

            if done:

                # reset recurrent memories
                for drone_idx in range(NUM_DRONES):

                    actor_hiddens[drone_idx] = \
                        agent.actor.init_hidden(1)

                    critic_hiddens[drone_idx] = \
                        agent.critic.init_hidden(1)

                break

        # ====================================================
        # LOGGING
        # ====================================================

        episode_rewards.append(episode_reward)

        print(
            f"Episode: {episode} | "
            f"Reward: {episode_reward:.2f}"
        )

        print(
            f"Coverage: {env.debug_stats['coverage'][-1]:.3f} | "
            f"EscapeGap: {env.debug_stats['escape_gap'][-1]:.3f} | "
            f"MeanDist: {env.debug_stats['mean_distance'][-1]:.3f}"
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
                target_positions=env.target_history,
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

            plot_capture_rate(
                env.debug_stats["capture_rate"],
                save_path="plots/capture_rate.png"
            )

            plot_mean_distance(
                env.debug_stats["mean_distance"],
                save_path="plots/mean_distance.png"
            )

            plot_coverage(
                env.debug_stats["coverage"],
                save_path="plots/coverage.png"
            )

            plot_escape_gap(
                env.debug_stats["escape_gap"],
                save_path="plots/escape_gap.png"
            )

            plot_velocity_diversity(
                env.debug_stats["velocity_diversity"],
                save_path="plots/velocity_diversity.png"
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

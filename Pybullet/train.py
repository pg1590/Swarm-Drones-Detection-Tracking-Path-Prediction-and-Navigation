from pybullet_env import PyBulletPursuitEnv

from mappo_agent import MAPPO

from rollout_buffer import RolloutBuffer

import os
import csv
import json
import torch
import numpy as np


# ====================================================
# TRAINING HYPERPARAMETERS
# ====================================================

MAX_EPISODES = 5000

ROLLOUT_STEPS = 2048

PPO_EPOCHS = 6

MINI_BATCH_SIZE = 256

SAVE_MODEL_EVERY = 500

SAVE_TRAJECTORY_EVERY = 100


# ====================================================
# DIRECTORIES
# ====================================================

os.makedirs("logs", exist_ok=True)

os.makedirs("models", exist_ok=True)

os.makedirs("trajectories", exist_ok=True)


# ====================================================
# CSV LOGGER
# ====================================================

log_file = open(
    "logs/training_log.csv",
    "w",
    newline=""
)

csv_writer = csv.writer(log_file)

csv_writer.writerow([

    "episode",

    "episode_reward",

    "capture",

    "capture_step",

    "episode_length",

    "avg_distance",

    "avg_inter_agent_dist",

    "avg_enclosure",

    "success_rate"

])


# ====================================================
# MAIN TRAINING LOOP
# ====================================================

def main():

    # ------------------------------------------------
    # Environment
    # ------------------------------------------------

    env = PyBulletPursuitEnv(gui=False)

    # ------------------------------------------------
    # Dimensions
    # ------------------------------------------------

    obs_dim = len(env.reset()["p1"])

    global_state_dim = len(
        env.get_global_state()
    )

    action_dim = 2

    # ------------------------------------------------
    # MAPPO Agent
    # ------------------------------------------------

    agent = MAPPO(

        obs_dim=obs_dim,

        global_state_dim=global_state_dim,

        action_dim=action_dim
    )

    # ------------------------------------------------
    # Rollout Buffer
    # ------------------------------------------------

    buffer = RolloutBuffer()

    success_count = 0

    total_steps = 0

    # =================================================
    # EPISODES
    # =================================================

    for episode in range(MAX_EPISODES):

        observations = env.reset()

        done = False

        episode_reward = 0

        trajectory = []

        # ------------------------------------------------
        # Metrics
        # ------------------------------------------------

        distance_sum = 0.0

        inter_agent_sum = 0.0

        enclosure_sum = 0.0

        capture_step = -1

        # =================================================
        # TIMESTEP LOOP
        # =================================================

        while not done:

            # --------------------------------------------
            # Local observations
            # --------------------------------------------

            obs1 = observations["p1"]

            obs2 = observations["p2"]

            # --------------------------------------------
            # Global critic state
            # --------------------------------------------

            global_state = env.get_global_state()

            # --------------------------------------------
            # Shared policy actions
            # --------------------------------------------

            action1, logprob1 = (
                agent.select_action(obs1)
            )

            action2, logprob2 = (
                agent.select_action(obs2)
            )

            # --------------------------------------------
            # Critic value
            # --------------------------------------------

            with torch.no_grad():

                value = agent.critic(

                    torch.FloatTensor(
                        global_state
                    ).unsqueeze(0)

                ).item()

            # --------------------------------------------
            # Environment step
            # --------------------------------------------

            next_observations, reward, done = env.step(

                action1,

                action2
            )

            # --------------------------------------------
            # Store rollout
            # --------------------------------------------

            buffer.store(

                obs1,

                global_state,

                action1,

                logprob1,

                reward,

                done,

                value
            )

            buffer.store(

                obs2,

                global_state,

                action2,

                logprob2,

                reward,

                done,

                value
            )

            # --------------------------------------------
            # Metrics
            # --------------------------------------------

            d1 = np.linalg.norm(
                env.p1_pos - env.evader_pos
            )

            d2 = np.linalg.norm(
                env.p2_pos - env.evader_pos
            )

            avg_dist = (d1 + d2) / 2

            distance_sum += avg_dist

            inter_agent_dist = np.linalg.norm(
                env.p1_pos - env.p2_pos
            )

            inter_agent_sum += inter_agent_dist

            # --------------------------------------------
            # Enclosure geometry
            # --------------------------------------------

            v1 = env.p1_pos - env.evader_pos
            v2 = env.p2_pos - env.evader_pos

            v1 /= np.linalg.norm(v1) + 1e-6
            v2 /= np.linalg.norm(v2) + 1e-6

            dot = np.clip(
                np.dot(v1, v2),
                -1,
                1
            )

            angle = np.arccos(dot) / np.pi

            enclosure_sum += angle

            # --------------------------------------------
            # Capture timestep
            # --------------------------------------------

            if env.capture and capture_step == -1:

                capture_step = env.step_count

            # --------------------------------------------
            # Trajectory logging
            # --------------------------------------------

            trajectory.append({

                "p1_pos":
                    env.p1_pos.tolist(),

                "p2_pos":
                    env.p2_pos.tolist(),

                "evader_pos":
                    env.evader_pos.tolist(),

                "p1_vel":
                    env.p1_vel.tolist(),

                "p2_vel":
                    env.p2_vel.tolist(),

                "evader_vel":
                    env.evader_vel.tolist(),

                "reward":
                    float(reward)
            })

            # --------------------------------------------
            # Update
            # --------------------------------------------

            observations = next_observations

            episode_reward += reward

            total_steps += 1

            # =================================================
            # PPO UPDATE
            # =================================================

            if total_steps % ROLLOUT_STEPS == 0:

                data = buffer.get()

                # --------------------------------------------
                # Bootstrap value
                # --------------------------------------------

                with torch.no_grad():

                    next_global_state = (
                        env.get_global_state()
                    )

                    next_value = agent.critic(

                        torch.FloatTensor(
                            next_global_state
                        ).unsqueeze(0)

                    ).item()

                # --------------------------------------------
                # GAE
                # --------------------------------------------

                advantages, returns = (

                    agent.compute_gae(

                        rewards=data["rewards"],

                        values=list(data["values"]),

                        dones=data["dones"],

                        next_value=next_value
                    )
                )

                # --------------------------------------------
                # PPO update
                # --------------------------------------------

                agent.update(

                    obs_batch=data["obs"],

                    global_state_batch=data[
                        "global_states"
                    ],

                    action_batch=data[
                        "actions"
                    ],

                    old_logprob_batch=data[
                        "log_probs"
                    ],

                    returns_batch=returns,

                    advantage_batch=advantages,

                    epochs=PPO_EPOCHS,

                    batch_size=MINI_BATCH_SIZE
                )

                buffer.clear()

        # =================================================
        # END EPISODE
        # =================================================

        if env.capture:

            success_count += 1

        episode_length = env.step_count

        avg_distance = (
            distance_sum / episode_length
        )

        avg_inter_agent_dist = (
            inter_agent_sum / episode_length
        )

        avg_enclosure = (
            enclosure_sum / episode_length
        )

        success_rate = (
            success_count / (episode + 1)
        )

        # =================================================
        # CSV LOGGING
        # =================================================

        csv_writer.writerow([

            episode,

            episode_reward,

            int(env.capture),

            capture_step,

            episode_length,

            avg_distance,

            avg_inter_agent_dist,

            avg_enclosure,

            success_rate
        ])

        log_file.flush()

        # =================================================
        # SAVE TRAJECTORIES
        # =================================================

        if episode % SAVE_TRAJECTORY_EVERY == 0:

            with open(

                f"trajectories/ep_{episode}.json",

                "w"

            ) as f:

                json.dump(
                    trajectory,
                    f
                )

        # =================================================
        # SAVE MODELS
        # =================================================

        if episode % SAVE_MODEL_EVERY == 0:

            torch.save(

                agent.actor.state_dict(),

                f"models/actor_{episode}.pth"
            )

            torch.save(

                agent.critic.state_dict(),

                f"models/critic_{episode}.pth"
            )

        # =================================================
        # TERMINAL LOGGING
        # =================================================

        if episode % 20 == 0:

            print("=" * 50)

            print(f"Episode: {episode}")

            print(
                f"Reward: "
                f"{episode_reward:.2f}"
            )

            print(
                f"Capture: "
                f"{env.capture}"
            )

            print(
                f"Capture Step: "
                f"{capture_step}"
            )

            print(
                f"Avg Distance: "
                f"{avg_distance:.2f}"
            )

            print(
                f"Avg Inter-Agent Dist: "
                f"{avg_inter_agent_dist:.2f}"
            )

            print(
                f"Avg Enclosure: "
                f"{avg_enclosure:.3f}"
            )

            print(
                f"Success Rate: "
                f"{success_rate:.3f}"
            )

            print(
                f"Total Steps: "
                f"{total_steps}"
            )

            print("=" * 50)

    print("Training Complete")

    log_file.close()


# ====================================================
# ENTRY
# ====================================================

if __name__ == "__main__":

    main()
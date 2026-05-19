from env import PursuitEnv

from mappo_agent import MAPPO

from rollout_buffer import RolloutBuffer
from utils import compute_swarm_metrics
import torch
import numpy as np
import logging
import os
from datetime import datetime

# -----------------------------------
# Logging Setup
# -----------------------------------

os.makedirs("logs", exist_ok=True)

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

log_filename = f"logs/mappo_{timestamp}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

logger = logging.getLogger()

# ------------------------------------------------
# Training Hyperparameters
# ------------------------------------------------

MAX_EPISODES = 5000

ROLLOUT_STEPS = 4096

PPO_EPOCHS = 6

MINI_BATCH_SIZE = 512


def main():

    env = PursuitEnv()

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
        action_dim=action_dim,logger=logger
    )

    # ------------------------------------------------
    # Rollout Buffer
    # ------------------------------------------------

    buffer = RolloutBuffer()

    success_count = 0
    min_evader_dist = 1e9

    total_steps = 0

    for episode in range(MAX_EPISODES):
        episode_metrics = {
            "separation": [],
            "enclosure_angle": [],
            "mean_evader_dist": [],
            "team_center_dist": []
        }
        persistent_trap_steps = 0
        observations = env.reset()

        done = False

        episode_reward = 0

        while not done:

            # ----------------------------------------
            # Local observations
            # ----------------------------------------

            obs1 = observations["p1"]

            obs2 = observations["p2"]

            # ----------------------------------------
            # Centralized critic state
            # ----------------------------------------

            global_state = env.get_global_state()

            # ----------------------------------------
            # Shared Actor Actions
            # ----------------------------------------

            action1, logprob1 = (
                agent.select_action(obs1)
            )

            action2, logprob2 = (
                agent.select_action(obs2)
            )

            # ----------------------------------------
            # Critic Value
            # ----------------------------------------

            with torch.no_grad():

                value = agent.critic(
                    torch.FloatTensor(
                        global_state
                    ).unsqueeze(0)
                ).item()

            # ----------------------------------------
            # Environment step
            # ----------------------------------------

            next_observations, reward, done = env.step(
                action1,
                action2
            )
            metrics = compute_swarm_metrics(
                env.pursuer1_pos,
                env.pursuer2_pos,
                env.evader_pos
            )
            if metrics["enclosure_angle"] > 120:
                persistent_trap_steps += 1
            for key in episode_metrics:
                episode_metrics[key].append(
                    metrics[key]
                )
            min_evader_dist = min(
                min_evader_dist,
                metrics["mean_evader_dist"]
            )
            # ----------------------------------------
            # Store BOTH agents separately
            # ----------------------------------------

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

            observations = next_observations

            episode_reward += reward

            total_steps += 1

            # ----------------------------------------
            # PPO Update
            # ----------------------------------------
            if total_steps % ROLLOUT_STEPS == 0:

                data = buffer.get()

                # ------------------------------
                # Bootstrap next state value
                # ------------------------------

                with torch.no_grad():

                    next_global_state = (
                        env.get_global_state()
                    )

                    next_value = agent.critic(
                        torch.FloatTensor(
                            next_global_state
                        ).unsqueeze(0)
                    ).item()

                # ------------------------------
                # Compute GAE
                # ------------------------------

                advantages, returns = (
                    agent.compute_gae(
                        rewards=data["rewards"],
                        values=list(data["values"]),
                        dones=data["dones"],
                        next_value=next_value
                    )
                )

                # ------------------------------
                # PPO Update
                # ------------------------------

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

        # ----------------------------------------
        # Success tracking
        # ----------------------------------------

        if env.capture:
            success_count += 1

        # ----------------------------------------
        # Logging
        # ----------------------------------------

        if episode % 50 == 0:

            print("=" * 50)

            print(f"Episode: {episode}")

            print(
                f"Reward: {episode_reward:.2f}"
            )

            print(
                f"Success Rate: "
                f"{success_count/(episode+1):.3f}"
            )

            print(
                f"Total Steps: {total_steps}"
            )

            print("=" * 50)
            logger.info(
                f"Mean Separation: "
                f"{np.mean(episode_metrics['separation']):.3f}"
            )

            logger.info(
                f"Mean Enclosure Angle: "
                f"{np.mean(episode_metrics['enclosure_angle']):.3f}"
            )

            logger.info(
                f"Mean Evader Distance: "
                f"{np.mean(episode_metrics['mean_evader_dist']):.3f}"
            )

            logger.info(
                f"Mean Team Center Dist: "
                f"{np.mean(episode_metrics['team_center_dist']):.3f}"
            )
            logger.info(
                f"Trap Persistence Steps: "
                f"{persistent_trap_steps/ max(1, env.steps):.3f}"
            )
            logger.info(
                f"Min Evader Dist: "
                f"{min_evader_dist:.3f}"
            )

    print("Training Complete")


if __name__ == "__main__":

    main()
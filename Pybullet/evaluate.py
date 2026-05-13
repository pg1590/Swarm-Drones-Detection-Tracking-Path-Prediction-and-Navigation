import torch
import numpy as np

from pybullet_env import PyBulletPursuitEnv
from mappo_agent import MAPPO


def main():

    # ------------------------------------------------
    # GUI ENV
    # ------------------------------------------------

    env = PyBulletPursuitEnv(gui=True)

    # ------------------------------------------------
    # Dimensions
    # ------------------------------------------------

    obs_dim = len(env.reset()["p1"])

    global_state_dim = len(
        env.get_global_state()
    )

    action_dim = 2

    # ------------------------------------------------
    # Agent
    # ------------------------------------------------

    agent = MAPPO(

        obs_dim=obs_dim,

        global_state_dim=global_state_dim,

        action_dim=action_dim
    )

    # ------------------------------------------------
    # Load trained actor
    # ------------------------------------------------

    agent.actor.load_state_dict(

        torch.load(
            "models/actor_4500.pth",
            map_location=torch.device("cpu")
        )
    )

    agent.actor.eval()

    # =================================================
    # Evaluation episodes
    # =================================================

    for ep in range(30):

        observations = env.reset()

        done = False

        total_reward = 0

        while not done:

            obs1 = observations["p1"]

            obs2 = observations["p2"]

            # ----------------------------------------
            # Deterministic actions
            # ----------------------------------------

            with torch.no_grad():

                obs1_t = torch.FloatTensor(
                    obs1
                ).unsqueeze(0)

                obs2_t = torch.FloatTensor(
                    obs2
                ).unsqueeze(0)

                dist1 = agent.actor(obs1_t)

                dist2 = agent.actor(obs2_t)

                action1 = dist1.mean.squeeze(0).numpy()

                action2 = dist2.mean.squeeze(0).numpy()

            

            observations, reward, done = env.step(

                action1,

                action2
            )

            total_reward += reward

        print("=" * 50)

        print(f"Episode {ep}")

        print(f"Reward: {total_reward:.2f}")

        print(f"Capture: {env.capture}")

        print("=" * 50)

        print("Evaluation Complete")

        input("Press Enter to close...")


if __name__ == "__main__":

    main()
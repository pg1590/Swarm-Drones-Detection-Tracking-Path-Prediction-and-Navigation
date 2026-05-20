import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# TRAJECTORY PLOT
# ============================================================

def plot_trajectories(
    trajectory_history,
    target_positions=None,
    save_path=None,
    show_plot=False
):

    """
    trajectory_history:
        list of shape:
        [timesteps][num_agents][2]

    target_positions:
        optional list of target positions

    save_path:
        optional path to save figure

    show_plot:
        whether to display figure
    """

    trajectory_history = np.array(
        trajectory_history
    )

    num_steps = trajectory_history.shape[0]

    num_agents = trajectory_history.shape[1]

    plt.figure(figsize=(10, 10))

    # ========================================================
    # AGENT TRAJECTORIES
    # ========================================================

    for agent_idx in range(num_agents):

        traj = trajectory_history[:, agent_idx]

        x = traj[:, 0]
        y = traj[:, 1]

        # trajectory line
        plt.plot(
            x,
            y,
            linewidth=2,
            label=f"Drone {agent_idx}"
        )

        # start point
        plt.scatter(
            x[0],
            y[0],
            marker='o',
            s=100
        )

        # end point
        plt.scatter(
            x[-1],
            y[-1],
            marker='x',
            s=100
        )

    # ========================================================
    # TARGET TRAJECTORY
    # ========================================================

    if target_positions is not None:

        target_positions = np.array(
            target_positions
        )

        tx = target_positions[:, 0]
        ty = target_positions[:, 1]

        plt.plot(
            tx,
            ty,
            'k--',
            linewidth=3,
            label='Target'
        )

        plt.scatter(
            tx[0],
            ty[0],
            c='black',
            marker='s',
            s=120
        )

        plt.scatter(
            tx[-1],
            ty[-1],
            c='red',
            marker='*',
            s=150
        )

    # ========================================================
    # PLOT SETTINGS
    # ========================================================

    plt.title(
        "Drone Swarm Trajectories",
        fontsize=18
    )

    plt.xlabel("X Position")

    plt.ylabel("Y Position")

    plt.grid(True)

    plt.axis('equal')

    plt.legend()

    # ========================================================
    # SAVE
    # ========================================================

    if save_path is not None:

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        plt.savefig(
            save_path,
            bbox_inches='tight'
        )

        print(f"Trajectory plot saved to: {save_path}")

    # ========================================================
    # SHOW
    # ========================================================

    if show_plot:

        plt.show()

    plt.close()


# ============================================================
# REWARD CURVE
# ============================================================

def plot_rewards(
    rewards,
    save_path=None,
    show_plot=False
):

    rewards = np.array(rewards)

    plt.figure(figsize=(10, 5))

    plt.plot(rewards)

    plt.title("Training Rewards")

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.grid(True)

    if save_path is not None:

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        plt.savefig(
            save_path,
            bbox_inches='tight'
        )

        print(f"Reward plot saved to: {save_path}")

    if show_plot:

        plt.show()

    plt.close()


# ============================================================
# SMOOTH REWARD CURVE
# ============================================================

def plot_smoothed_rewards(
    rewards,
    window=20,
    save_path=None,
    show_plot=False
):

    rewards = np.array(rewards)

    smoothed = np.convolve(
        rewards,
        np.ones(window) / window,
        mode='valid'
    )

    plt.figure(figsize=(10, 5))

    plt.plot(smoothed)

    plt.title(
        f"Smoothed Rewards (window={window})"
    )

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.grid(True)

    if save_path is not None:

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        plt.savefig(
            save_path,
            bbox_inches='tight'
        )

        print(f"Smoothed reward plot saved to: {save_path}")

    if show_plot:

        plt.show()

    plt.close()
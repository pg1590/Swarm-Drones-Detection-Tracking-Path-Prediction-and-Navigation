import numpy as np
import matplotlib.pyplot as plt
import os


def _ensure_save_dir(save_path):

    if save_path is None:
        return

    directory = os.path.dirname(save_path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )


# ============================================================
# TRAJECTORY PLOT
# ============================================================

def plot_trajectories(
    trajectory_history,
    target_positions=None,
    save_path=None,
    show_plot=False,
    show_escape_corridor=True
):

    """
    trajectory_history:
        [timesteps][num_agents][2]

    target_positions:
        [timesteps][2]
    """

    trajectory_history = np.array(
        trajectory_history
    )

    num_steps = trajectory_history.shape[0]

    num_agents = trajectory_history.shape[1]

    plt.figure(figsize=(10, 10))

    # ========================================================
    # DRONE TRAJECTORIES
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
            s=120
        )

        # end point
        plt.scatter(
            x[-1],
            y[-1],
            marker='x',
            s=120
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

        # target start
        plt.scatter(
            tx[0],
            ty[0],
            c='black',
            marker='s',
            s=180,
            label='Target Start'
        )

        # target end
        plt.scatter(
            tx[-1],
            ty[-1],
            c='red',
            marker='*',
            s=220,
            label='Target End'
        )

    # ========================================================
    # ESCAPE CORRIDOR VISUALIZATION
    # ========================================================

    if (
        show_escape_corridor
        and target_positions is not None
    ):

        final_target = target_positions[-1]

        final_positions = trajectory_history[-1]

        angles = []

        for pos in final_positions:

            vec = pos - final_target

            angle = np.arctan2(
                vec[1],
                vec[0]
            )

            angles.append(angle)

        angles = np.sort(angles)

        # largest angular gap
        largest_gap = 0
        best_pair = None

        for i in range(len(angles) - 1):

            gap = angles[i + 1] - angles[i]

            if gap > largest_gap:

                largest_gap = gap

                best_pair = (
                    angles[i],
                    angles[i + 1]
                )

        wrap_gap = (
            2 * np.pi
            - angles[-1]
            + angles[0]
        )

        if wrap_gap > largest_gap:

            largest_gap = wrap_gap

            best_pair = (
                angles[-1],
                angles[0] + 2 * np.pi
            )

        # visualize corridor
        theta_mid = (
            best_pair[0]
            + best_pair[1]
        ) / 2

        corridor_length = 5

        dx = corridor_length * np.cos(theta_mid)
        dy = corridor_length * np.sin(theta_mid)

        plt.arrow(
            final_target[0],
            final_target[1],
            dx,
            dy,
            width=0.025,
            head_width=0.22,
            color='crimson',
            alpha=0.45,
            linestyle='--',
            length_includes_head=True,
            label='Largest Escape Corridor',
            zorder=1
        )

    # ========================================================
    # FINAL SWARM CONNECTIONS
    # ========================================================

    final_positions = trajectory_history[-1]

    for i in range(num_agents):

        for j in range(i + 1, num_agents):

            p1 = final_positions[i]
            p2 = final_positions[j]

            plt.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                alpha=0.3
            )

    # ========================================================
    # PLOT SETTINGS
    # ========================================================

    plt.title(
        "Drone Swarm Trajectories",
        fontsize=20
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

        _ensure_save_dir(save_path)

        plt.savefig(
            save_path,
            bbox_inches='tight'
        )

        print(
            f"Trajectory plot saved to: {save_path}"
        )

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

        _ensure_save_dir(save_path)

        plt.savefig(
            save_path,
            bbox_inches='tight'
        )

        print(
            f"Reward plot saved to: {save_path}"
        )

    if show_plot:

        plt.show()

    plt.close()


# ============================================================
# SMOOTHED REWARD CURVE
# ============================================================

def plot_smoothed_rewards(
    rewards,
    window=20,
    save_path=None,
    show_plot=False
):

    rewards = np.array(rewards)

    if len(rewards) > 0:
        window = min(window, len(rewards))

        smoothed = np.convolve(
            rewards,
            np.ones(window) / window,
            mode='valid'
        )
    else:
        smoothed = np.array([])

    plt.figure(figsize=(10, 5))

    plt.plot(smoothed)

    plt.title(
        f"Smoothed Rewards (window={window})"
    )

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.grid(True)

    if save_path is not None:

        _ensure_save_dir(save_path)

        plt.savefig(
            save_path,
            bbox_inches='tight'
        )

        print(
            f"Smoothed reward plot saved to: {save_path}"
        )

    if show_plot:

        plt.show()

    plt.close()

def plot_capture_rate(
    capture_history,
    window=50,
    save_path=None
):

    capture_history = np.array(
        capture_history
    )

    if len(capture_history) > 0:
        window = min(window, len(capture_history))

        smoothed = np.convolve(
            capture_history,
            np.ones(window) / window,
            mode='valid'
        )
    else:
        smoothed = np.array([])

    plt.figure(figsize=(10, 5))

    plt.plot(smoothed)

    plt.title(
        "Capture Rate"
    )

    plt.xlabel("Episode")

    plt.ylabel("Capture Probability")

    plt.grid(True)

    if save_path is not None:

        _ensure_save_dir(save_path)

        plt.savefig(
            save_path,
            bbox_inches='tight'
        )

    plt.close()


def plot_mean_distance(
    distance_history,
    save_path=None
):

    plt.figure(figsize=(10, 5))

    plt.plot(distance_history)

    plt.title(
        "Mean Drone-Target Distance"
    )

    plt.xlabel("Episode")

    plt.ylabel("Distance")

    plt.grid(True)

    if save_path is not None:

        _ensure_save_dir(save_path)

        plt.savefig(
            save_path,
            bbox_inches='tight'
        )

    plt.close()


def plot_coverage(
    coverage_history,
    save_path=None
):

    plt.figure(figsize=(10, 5))

    plt.plot(coverage_history)

    plt.title(
        "Angular Coverage Reward"
    )

    plt.xlabel("Episode")

    plt.ylabel("Coverage")

    plt.grid(True)

    if save_path is not None:

        _ensure_save_dir(save_path)

        plt.savefig(
            save_path,
            bbox_inches='tight'
        )

    plt.close()


def plot_escape_gap(
    gap_history,
    save_path=None
):

    plt.figure(figsize=(10, 5))

    plt.plot(gap_history)

    plt.title(
        "Largest Escape Corridor"
    )

    plt.xlabel("Episode")

    plt.ylabel("Angular Gap (rad)")

    plt.grid(True)

    if save_path is not None:

        _ensure_save_dir(save_path)

        plt.savefig(
            save_path,
            bbox_inches='tight'
        )

    plt.close()


def plot_metric(
    metric_history,
    title,
    ylabel,
    save_path=None
):

    plt.figure(figsize=(10, 5))

    plt.plot(metric_history)

    plt.title(title)

    plt.xlabel("Step")

    plt.ylabel(ylabel)

    plt.grid(True)

    if save_path is not None:

        _ensure_save_dir(save_path)

        plt.savefig(
            save_path,
            bbox_inches='tight'
        )

    plt.close()

def plot_velocity_diversity(
    diversity_history,
    save_path=None
):

    plt.figure(figsize=(10, 5))

    plt.plot(diversity_history)

    plt.title(
        "Velocity Diversity"
    )

    plt.xlabel("Episode")

    plt.ylabel("Diversity")

    plt.grid(True)

    if save_path is not None:

        _ensure_save_dir(save_path)

        plt.savefig(
            save_path,
            bbox_inches='tight'
        )

    plt.close()

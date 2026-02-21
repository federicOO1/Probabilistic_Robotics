import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_trajectory(estimated_poses, gt_poses, save_path):

    ensure_dir(save_path)

    est_positions = np.array([T[:3, 3] for T in estimated_poses])
    gt_positions = np.array([T[:3, 3] for T in gt_poses])

    plt.figure()
    plt.plot(est_positions[:, 0], est_positions[:, 1])
    plt.plot(gt_positions[:, 0], gt_positions[:, 1])

    plt.title("Trajectory (XY)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(["Estimated", "Ground Truth"])
    plt.axis("equal")

    plt.savefig(os.path.join(save_path, "trajectory.png"))
    plt.close()


def plot_scale_ratio(scale_series, save_path):

    ensure_dir(save_path)

    plt.figure()
    plt.plot(scale_series)
    plt.title("Scale Ratio over Time")
    plt.xlabel("Frame")
    plt.ylabel("Scale Ratio")

    plt.savefig(os.path.join(save_path, "scale_ratio.png"))
    plt.close()


def plot_map(estimated_landmarks, gt_landmarks, scale_ratio, save_path):

    from mpl_toolkits.mplot3d import Axes3D

    ensure_dir(save_path)

    est_points = []
    gt_points = []

    scale = 1.0 / scale_ratio

    for landmark_id, X_est in estimated_landmarks.items():
        if landmark_id in gt_landmarks:
            est_points.append(scale * X_est)
            gt_points.append(gt_landmarks[landmark_id])

    est_points = np.array(est_points)
    gt_points = np.array(gt_points)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(est_points[:, 0], est_points[:, 1], est_points[:, 2])
    ax.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2])

    ax.set_title("3D Map (Estimated vs GT)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.savefig(os.path.join(save_path, "map_3d.png"))
    plt.close()

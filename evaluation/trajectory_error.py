import numpy as np
from geometry.se3 import se3_inverse


def pose2d_to_se3(x, y, theta):
    """
    Converts planar pose (x, y, theta) to SE(3).
    """
    T = np.eye(4)

    c = np.cos(theta)
    s = np.sin(theta)

    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c

    T[0, 3] = x
    T[1, 3] = y

    return T


def load_groundtruth(path):
    """
    Loads trajectory.dat
    Returns list of SE(3) GT poses.
    """
    gt_poses = []

    with open(path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))

            x_gt = values[4]
            y_gt = values[5]
            theta_gt = values[6]

            T_gt = pose2d_to_se3(x_gt, y_gt, theta_gt)
            gt_poses.append(T_gt)

    return gt_poses


def compute_relative_transform(T1, T2):
    return se3_inverse(T1) @ T2


def evaluate_trajectory(estimated_poses, gt_poses):
    """
    Computes relative pose errors.
    """

    rot_errors = []
    trans_ratios = []

    for i in range(len(estimated_poses) - 1):

        rel_est = compute_relative_transform(
            estimated_poses[i],
            estimated_poses[i + 1]
        )

        rel_gt = compute_relative_transform(
            gt_poses[i],
            gt_poses[i + 1]
        )

        error = se3_inverse(rel_est) @ rel_gt

        # Rotation error
        R_error = error[:3, :3]
        rot_error = np.trace(np.eye(3) - R_error)
        rot_errors.append(rot_error)

        # Translation ratio (scale consistency)
        t_est = rel_est[:3, 3]
        t_gt = rel_gt[:3, 3]

        norm_est = np.linalg.norm(t_est)
        norm_gt = np.linalg.norm(t_gt)

        if norm_gt > 1e-3:
            trans_ratios.append(norm_est / norm_gt)

    print("Scale ratios sample:", trans_ratios[:10])

    return np.mean(rot_errors), np.mean(trans_ratios), trans_ratios

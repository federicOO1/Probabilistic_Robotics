import numpy as np
import cv2


def normalize_points(K, pts):
    """
    Converts pixel coordinates to normalized camera coordinates.
    """
    K_inv = np.linalg.inv(K)

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (K_inv @ pts_h.T).T

    return pts_norm[:, :2]


def estimate_essential_matrix(K, pts1, pts2):
    """
    Estimates Essential matrix using OpenCV.
    """

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    return E, mask


def recover_pose_from_essential(K, E, pts1, pts2):
    """
    Recovers R and t from Essential matrix.
    """

    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    return R, t, mask


def triangulate_points(K, T0, T1, pts0, pts1):
    """
    Triangulates 3D points from two views.
    """

    P0 = K @ T0[:3]
    P1 = K @ T1[:3]

    pts4d = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)

    pts3d = pts4d[:3] / pts4d[3]

    return pts3d.T


def initialize_two_view(K, pts0, pts1):
    """
    Full two-view initialization pipeline.

    Returns:
        T0, T1, points_3d
    """

    T0 = np.eye(4)

    E, mask = estimate_essential_matrix(K, pts0, pts1)

    R, t, mask_pose = recover_pose_from_essential(K, E, pts0, pts1)

    # Normalize translation (fix scale)
    t = t / np.linalg.norm(t)

    T1 = np.eye(4)
    T1[:3, :3] = R
    T1[:3, 3] = t.flatten()

    points_3d = triangulate_points(K, T0, T1, pts0, pts1)

    return T0, T1, points_3d

import numpy as np
import cv2


def build_projection_matrix(K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Builds projection matrix P = K [R | t]

    T: 4x4 pose matrix (world to camera)
    Returns:
        3x4 projection matrix
    """
    return K @ T[:3]


def triangulate_point(
    K: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    pt1: np.ndarray,
    pt2: np.ndarray
) -> np.ndarray:
    """
    Triangulates a single 3D point from two views.

    pt1, pt2: 2D image coordinates (u, v)
    """

    P1 = build_projection_matrix(K, T1)
    P2 = build_projection_matrix(K, T2)

    pt1_h = pt1.reshape(2, 1)
    pt2_h = pt2.reshape(2, 1)

    X_h = cv2.triangulatePoints(P1, P2, pt1_h, pt2_h)

    X = X_h[:3] / X_h[3]

    return X.flatten()


def triangulate_points_batch(
    K: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray
) -> np.ndarray:
    """
    Triangulates multiple points at once.

    pts1, pts2: Nx2 arrays
    Returns:
        Nx3 array of 3D points
    """

    P1 = build_projection_matrix(K, T1)
    P2 = build_projection_matrix(K, T2)

    pts1 = pts1.T
    pts2 = pts2.T

    X_h = cv2.triangulatePoints(P1, P2, pts1, pts2)
    X = X_h[:3] / X_h[3]

    return X.T

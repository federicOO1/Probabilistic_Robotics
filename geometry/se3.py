import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """
    Returns the skew-symmetric matrix of a 3D vector.
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """
    Exponential map for SO(3) using Rodrigues' formula.

    omega: 3D rotation vector (axis-angle representation)
    Returns: 3x3 rotation matrix
    """
    theta = np.linalg.norm(omega)

    if theta < 1e-10:
        return np.eye(3)

    omega_hat = skew(omega / theta)

    R = (
        np.eye(3)
        + np.sin(theta) * omega_hat
        + (1 - np.cos(theta)) * (omega_hat @ omega_hat)
    )

    return R


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """
    Exponential map for SE(3).

    xi: 6D vector [rho, omega]
        rho   -> translation (3,)
        omega -> rotation (3,)

    Returns:
        4x4 homogeneous transformation matrix
    """
    rho = xi[:3]
    omega = xi[3:]

    theta = np.linalg.norm(omega)

    T = np.eye(4)

    if theta < 1e-10:
        R = np.eye(3)
        V = np.eye(3)
    else:
        omega_hat = skew(omega / theta)

        R = (
            np.eye(3)
            + np.sin(theta) * omega_hat
            + (1 - np.cos(theta)) * (omega_hat @ omega_hat)
        )

        V = (
            np.eye(3)
            + ((1 - np.cos(theta)) / theta) * omega_hat
            + ((theta - np.sin(theta)) / theta) * (omega_hat @ omega_hat)
        )

    t = V @ rho

    T[:3, :3] = R
    T[:3, 3] = t

    return T


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a 4x4 SE(3) matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def se3_compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    Composition of two SE(3) transformations.
    """
    return T1 @ T2


def transform_point(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Applies SE(3) transformation to a 3D point.

    X: 3D point (3,)
    Returns transformed 3D point (3,)
    """
    X_h = np.ones(4)
    X_h[:3] = X

    X_trans = T @ X_h

    return X_trans[:3]

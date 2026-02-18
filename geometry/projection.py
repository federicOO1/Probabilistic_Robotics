import numpy as np


def project_point(K: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Projects a 3D point (in camera frame) into image coordinates.

    K: camera intrinsic matrix (3x3)
    X: 3D point in camera frame (3,)

    Returns:
        2D image coordinates (u, v)
    """
    x, y, z = X

    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]

    return np.array([u, v])


def projection_jacobian(K: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Computes the Jacobian of the projection function wrt the 3D point X.

    Returns:
        2x3 Jacobian matrix
    """
    fx = K[0, 0]
    fy = K[1, 1]

    x, y, z = X

    J = np.array([
        [fx / z, 0, -fx * x / (z * z)],
        [0, fy / z, -fy * y / (z * z)]
    ])

    return J

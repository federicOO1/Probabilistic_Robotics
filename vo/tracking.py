import numpy as np
from geometry.se3 import skew, se3_exp, transform_point
from geometry.projection import project_point, projection_jacobian


def se3_point_jacobian(X_c):

    J = np.zeros((3, 6))
    J[:, :3] = np.eye(3)
    J[:, 3:] = -skew(X_c)
    return J


def compute_residual_and_jacobian(T, K, X_w, z):

    X_c = transform_point(T, X_w)
    z_hat = project_point(K, X_c)
    r = z - z_hat

    J_proj = projection_jacobian(K, X_c)
    J_se3 = se3_point_jacobian(X_c)

    J = J_proj @ J_se3

    return r, J


def gauss_newton_pose_estimation(
    T_init,
    K,
    points_3d,
    points_2d,
    max_iterations=10,
    tolerance=1e-6
):

    T = T_init.copy()

    for _ in range(max_iterations):

        H = np.zeros((6, 6))
        b = np.zeros(6)

        for X_w, z in zip(points_3d, points_2d):

            r, J = compute_residual_and_jacobian(T, K, X_w, z)

            H += J.T @ J
            b += J.T @ r

        lambda_damping = 1e-3
        H_damped = H + lambda_damping * np.eye(6)

        try:
            delta = np.linalg.solve(H_damped, b)
        except np.linalg.LinAlgError:
            print("Singular matrix")
            break

        T = se3_exp(delta) @ T

        if np.linalg.norm(delta) < tolerance:
            break

    return T

import numpy as np


def load_world_map(path):
    """
    Loads world.dat.

    Returns:
        dict {landmark_id: 3D position}
    """

    gt_landmarks = {}

    with open(path, "r") as f:
        for line in f:

            values = line.strip().split()

            if len(values) < 4:
                continue

            landmark_id = int(values[0])

            x = float(values[1])
            y = float(values[2])
            z = float(values[3])

            gt_landmarks[landmark_id] = np.array([x, y, z])

    return gt_landmarks


def evaluate_map(estimated_landmarks, gt_landmarks, scale_ratio):
    """
    Computes RMSE between estimated map and groundtruth map.

    estimated_landmarks: dict {id: 3D point}
    gt_landmarks: dict {id: 3D point}
    scale_ratio: scalar

    Returns:
        RMSE
    """

    errors = []

    scale = 1.0 / scale_ratio

    for landmark_id, X_est in estimated_landmarks.items():

        if landmark_id in gt_landmarks:

            X_scaled = scale * X_est
            X_gt = gt_landmarks[landmark_id]

            error = np.linalg.norm(X_scaled - X_gt)
            errors.append(error ** 2)

    if len(errors) == 0:
        return None

    rmse = np.sqrt(np.mean(errors))

    return rmse

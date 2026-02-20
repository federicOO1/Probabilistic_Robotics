import numpy as np
import os
import glob


def load_camera_intrinsics(path):
    with open(path, "r") as f:
        lines = f.readlines()

    K = []

    for i, line in enumerate(lines):
        if "camera matrix" in line.lower():
            for j in range(1, 4):
                row = list(map(float, lines[i + j].strip().split()))
                K.append(row)
            break

    if len(K) != 3:
        raise ValueError("Could not parse camera matrix")

    return np.array(K)


def load_measurement_file(path):

    keypoints = []
    descriptors = []
    actual_ids = []

    with open(path, "r") as f:
        for line in f:

            line = line.strip()

            if not line.startswith("point"):
                continue

            parts = line.split()

            actual_id = int(parts[2])
            u = float(parts[3])
            v = float(parts[4])
            desc = list(map(float, parts[5:15]))

            keypoints.append([u, v])
            descriptors.append(desc)
            actual_ids.append(actual_id)

    return (
        np.array(keypoints),
        np.array(descriptors),
        np.array(actual_ids)
    )


def load_all_measurements(data_folder):

    files = sorted(glob.glob(os.path.join(data_folder, "meas-*.dat")))

    frames = []

    for file in files:
        kpts, desc, ids = load_measurement_file(file)
        frames.append((kpts, desc, ids))

    return frames

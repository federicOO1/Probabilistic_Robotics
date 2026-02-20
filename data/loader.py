import numpy as np
import os
import glob


def load_camera_intrinsics(path):
    """
    Parses camera.dat and extracts intrinsic matrix K.
    """

    with open(path, "r") as f:
        lines = f.readlines()

    K = []

    for i, line in enumerate(lines):
        if "camera matrix" in line.lower():
            # Read the next 3 lines
            for j in range(1, 4):
                row = list(map(float, lines[i + j].strip().split()))
                K.append(row)
            break

    if len(K) != 3:
        raise ValueError("Could not parse camera matrix from camera.dat")

    return np.array(K)



def load_measurement_file(path):
    """
    Loads a meas-XXXX.dat file.

    Returns:
        keypoints: Nx2 array (u, v)
        descriptors: Nx10 array
    """

    keypoints = []
    descriptors = []

    with open(path, "r") as f:
        for line in f:

            line = line.strip()

            # Skip non-point lines
            if not line.startswith("point"):
                continue

            parts = line.split()

            # Extract values by correct indexing
            u = float(parts[3])
            v = float(parts[4])

            desc = list(map(float, parts[5:15]))

            keypoints.append([u, v])
            descriptors.append(desc)

    return np.array(keypoints), np.array(descriptors)


def load_all_measurements(data_folder):
    """
    Loads all meas-XXXX.dat files sorted by frame index.
    """

    files = sorted(glob.glob(os.path.join(data_folder, "meas-*.dat")))

    frames = []

    for file in files:
        kpts, desc = load_measurement_file(file)
        frames.append((kpts, desc))

    return frames

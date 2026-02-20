import numpy as np


def compute_l2_distance_matrix(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
    """
    Computes full pairwise L2 distance matrix between descriptors.

    desc1: NxD
    desc2: MxD

    Returns:
        NxM distance matrix
    """
    dists = np.linalg.norm(desc1[:, None, :] - desc2[None, :, :], axis=2)
    return dists


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    distance_threshold: float = 0.5,
    mutual_check: bool = True
):
    """
    Matches descriptors using nearest neighbor search.

    desc1: NxD
    desc2: MxD

    Returns:
        List of (index_in_desc1, index_in_desc2)
    """

    dists = compute_l2_distance_matrix(desc1, desc2)

    matches = []

    # Nearest neighbor from desc1 → desc2
    nn12 = np.argmin(dists, axis=1)
    min_dist12 = np.min(dists, axis=1)

    if mutual_check:
        # Nearest neighbor from desc2 → desc1
        nn21 = np.argmin(dists, axis=0)

        for i, j in enumerate(nn12):
            if min_dist12[i] < distance_threshold:
                if nn21[j] == i:
                    matches.append((i, j))
    else:
        for i, j in enumerate(nn12):
            if min_dist12[i] < distance_threshold:
                matches.append((i, j))

    return matches

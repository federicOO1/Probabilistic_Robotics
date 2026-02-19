import numpy as np

from vo.initialization import initialize_two_view
from vo.tracking import gauss_newton_pose_estimation
from vo.data_association import match_descriptors
from geometry.triangulation import triangulate_point
from geometry.se3 import transform_point


class VisualOdometry:

    def __init__(self, K):
        """
        K: camera intrinsic matrix (3x3)
        """
        self.K = K

        self.poses = []           # List of 4x4 SE(3) matrices
        self.landmarks = []       # List of 3D points (Nx3)

        self.initialized = False

        self.prev_keypoints = None
        self.prev_descriptors = None

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    def process_first_two_frames(self, kpts0, desc0, kpts1, desc1):
        """
        Performs two-view initialization using epipolar geometry.
        """

        matches = match_descriptors(desc0, desc1)

        if len(matches) < 8:
            print("Not enough matches for initialization")
            return

        pts0 = np.array([kpts0[i] for i, _ in matches])
        pts1 = np.array([kpts1[j] for _, j in matches])

        T0, T1, points_3d = initialize_two_view(self.K, pts0, pts1)

        self.poses.append(T0)
        self.poses.append(T1)

        self.landmarks = list(points_3d)

        self.prev_keypoints = kpts1
        self.prev_descriptors = desc1

        self.initialized = True

        print(f"Initialization complete with {len(self.landmarks)} landmarks")

    # --------------------------------------------------------
    # TRACKING + MAPPING
    # --------------------------------------------------------

    def process_frame(self, kpts, descriptors):
        """
        Processes a new frame after initialization.
        """

        if not self.initialized:
            raise RuntimeError("System not initialized")

        matches = match_descriptors(self.prev_descriptors, descriptors)

        if len(matches) < 6:
            print("Not enough matches for tracking")
            return

        points_3d = []
        points_2d = []

        # Build 3D-2D correspondences
        for idx_prev, idx_curr in matches:
            if idx_prev < len(self.landmarks):
                points_3d.append(self.landmarks[idx_prev])
                points_2d.append(kpts[idx_curr])

        if len(points_3d) < 6:
            print("Not enough valid 3D-2D correspondences")
            return

        points_3d = np.array(points_3d)
        points_2d = np.array(points_2d)

        T_init = self.poses[-1]

        # Pose estimation via Gauss-Newton
        T_new = gauss_newton_pose_estimation(
            T_init,
            self.K,
            points_3d,
            points_2d
        )

        self.poses.append(T_new)

        # Triangulate new landmarks
        self.triangulate_new_landmarks(
            self.prev_keypoints,
            kpts,
            matches,
            self.poses[-2],
            T_new
        )

        self.prev_keypoints = kpts
        self.prev_descriptors = descriptors

        print(f"Frame processed. Total landmarks: {len(self.landmarks)}")

    # --------------------------------------------------------
    # TRIANGULATION (Incremental Simple Version)
    # --------------------------------------------------------

    def triangulate_new_landmarks(
        self,
        prev_kpts,
        curr_kpts,
        matches,
        T_prev,
        T_curr
    ):
        """
        Triangulates new landmarks from unmatched correspondences.
        Simple version with positive depth check.
        """

        for idx_prev, idx_curr in matches:

            # If already triangulated, skip
            if idx_prev < len(self.landmarks):
                continue

            pt_prev = prev_kpts[idx_prev]
            pt_curr = curr_kpts[idx_curr]

            X = triangulate_point(
                self.K,
                T_prev,
                T_curr,
                pt_prev,
                pt_curr
            )

            # Cheirality check: point must be in front of both cameras
            X_cam_prev = transform_point(T_prev, X)
            X_cam_curr = transform_point(T_curr, X)

            if X_cam_prev[2] > 0 and X_cam_curr[2] > 0:
                self.landmarks.append(X)

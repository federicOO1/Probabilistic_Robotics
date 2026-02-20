import numpy as np

from vo.initialization import initialize_two_view
from vo.tracking import gauss_newton_pose_estimation
from vo.data_association import match_descriptors
from geometry.triangulation import triangulate_point
from geometry.se3 import transform_point


class VisualOdometry:

    def __init__(self, K):

        self.K = K

        self.poses = []
        self.landmarks = {}   # ACTUAL_ID → 3D point

        self.initialized = False

        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_ids = None

    # -------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------

    def process_first_two_frames(
        self, kpts0, desc0, ids0,
        kpts1, desc1, ids1
    ):

        matches = match_descriptors(desc0, desc1)

        if len(matches) < 8:
            print("Not enough matches for initialization")
            return

        pts0 = np.array([kpts0[i] for i, _ in matches])
        pts1 = np.array([kpts1[j] for _, j in matches])

        T0, T1, points_3d = initialize_two_view(self.K, pts0, pts1)

        self.poses.append(T0)
        self.poses.append(T1)

        # Store landmarks using ACTUAL_ID from first frame
        for (i, _), X in zip(matches, points_3d):
            actual_id = ids0[i]
            self.landmarks[actual_id] = X

        self.prev_keypoints = kpts1
        self.prev_descriptors = desc1
        self.prev_ids = ids1

        self.initialized = True

        print(f"Initialization complete with {len(self.landmarks)} landmarks")

    # -------------------------------------------------------
    # TRACKING
    # -------------------------------------------------------

    def process_frame(self, kpts, descriptors, ids):

        if not self.initialized:
            raise RuntimeError("System not initialized")

        matches = match_descriptors(self.prev_descriptors, descriptors)

        points_3d = []
        points_2d = []

        # Build correct 3D-2D correspondences
        for idx_prev, idx_curr in matches:

            actual_id = self.prev_ids[idx_prev]

            if actual_id in self.landmarks:
                points_3d.append(self.landmarks[actual_id])
                points_2d.append(kpts[idx_curr])

        if len(points_3d) < 6:
            print("Not enough correspondences")
            return

        points_3d = np.array(points_3d)
        points_2d = np.array(points_2d)

        T_init = self.poses[-1]

        T_new = gauss_newton_pose_estimation(
            T_init,
            self.K,
            points_3d,
            points_2d
        )

        self.poses.append(T_new)

        # Triangulate new landmarks
        for idx_prev, idx_curr in matches:

            actual_id = self.prev_ids[idx_prev]

            if actual_id not in self.landmarks:

                pt_prev = self.prev_keypoints[idx_prev]
                pt_curr = kpts[idx_curr]

                X = triangulate_point(
                    self.K,
                    self.poses[-2],
                    T_new,
                    pt_prev,
                    pt_curr
                )

                X_cam_prev = transform_point(self.poses[-2], X)
                X_cam_curr = transform_point(T_new, X)

                if X_cam_prev[2] > 0 and X_cam_curr[2] > 0:
                    self.landmarks[actual_id] = X

        self.prev_keypoints = kpts
        self.prev_descriptors = descriptors
        self.prev_ids = ids

        print(f"Frame processed. Total landmarks: {len(self.landmarks)}")

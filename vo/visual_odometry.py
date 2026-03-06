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
        self.landmarks = {}   # landmark_id -> 3D point

        self.next_landmark_id = 0

        self.initialized = False

        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_landmark_ids = None

    # -------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------

    def process_first_two_frames(
        self, kpts0, desc0,
        kpts1, desc1
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

        self.prev_landmark_ids = [None] * len(kpts1)

        for (idx0, idx1), X in zip(matches, points_3d):

            landmark_id = self.next_landmark_id
            self.next_landmark_id += 1

            self.landmarks[landmark_id] = X
            self.prev_landmark_ids[idx1] = landmark_id

        self.prev_keypoints = kpts1
        self.prev_descriptors = desc1

        self.initialized = True

        print(f"Initialization complete with {len(self.landmarks)} landmarks")

    # -------------------------------------------------------
    # TRACKING
    # -------------------------------------------------------

    def process_frame(self, kpts, descriptors):

        if not self.initialized:
            raise RuntimeError("System not initialized")

        matches = match_descriptors(self.prev_descriptors, descriptors)

        points_3d = []
        points_2d = []

        for idx_prev, idx_curr in matches:

            landmark_id = self.prev_landmark_ids[idx_prev]

            if landmark_id is not None:
                points_3d.append(self.landmarks[landmark_id])
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

        current_landmark_ids = [None] * len(kpts)

        for idx_prev, idx_curr in matches:

            landmark_id = self.prev_landmark_ids[idx_prev]

            if landmark_id is not None:
                current_landmark_ids[idx_curr] = landmark_id

            else:
                # triangulate new point
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

                    # triangulation angle filter
                    r1 = X_cam_prev / np.linalg.norm(X_cam_prev)
                    r2 = X_cam_curr / np.linalg.norm(X_cam_curr)

                    cos_angle = np.clip(np.dot(r1, r2), -1.0, 1.0)
                    angle = np.arccos(cos_angle)

                    min_angle = np.deg2rad(1.0)

                    if angle > min_angle:
                        landmark_id = self.next_landmark_id
                        self.next_landmark_id += 1

                        self.landmarks[landmark_id] = X
                        current_landmark_ids[idx_curr] = landmark_id

        self.prev_keypoints = kpts
        self.prev_descriptors = descriptors
        self.prev_landmark_ids = current_landmark_ids

        print(f"Frame processed. Total landmarks: {len(self.landmarks)}")
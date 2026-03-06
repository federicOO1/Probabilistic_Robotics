import os
from vo.visual_odometry import VisualOdometry
from data.loader import load_camera_intrinsics, load_all_measurements
from evaluation.trajectory_error import load_groundtruth, evaluate_trajectory
from evaluation.map_error import load_world_map, evaluate_map


def main():

    data_folder = "data"

    K = load_camera_intrinsics(os.path.join(data_folder, "camera.dat"))
    frames = load_all_measurements(data_folder)

    vo = VisualOdometry(K)

    # Initialization
    kpts0, desc0 = frames[0]
    kpts1, desc1 = frames[1]

    vo.process_first_two_frames(
        kpts0, desc0,
        kpts1, desc1
    )

    # Tracking
    for kpts, desc in frames[2:]:
        vo.process_frame(kpts, desc)

    print("VO finished.")
    print(f"Total poses: {len(vo.poses)}")
    print(f"Total landmarks: {len(vo.landmarks)}")

    gt_path = os.path.join(data_folder, "trajectory.dat")
    gt_poses = load_groundtruth(gt_path)

    rot_error, scale_ratio, scale_series = evaluate_trajectory(vo.poses, gt_poses)

    print("Mean rotation error:", rot_error)
    print("Mean scale ratio:", scale_ratio)

    world_path = os.path.join(data_folder, "world.dat")
    gt_landmarks = load_world_map(world_path)

    map_rmse = evaluate_map(vo.landmarks, gt_landmarks, scale_ratio)

    print("Map RMSE:", map_rmse)


if __name__ == "__main__":
    main()
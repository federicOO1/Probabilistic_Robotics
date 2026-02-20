import os
from vo.visual_odometry import VisualOdometry
from data.loader import load_camera_intrinsics, load_all_measurements


def main():

    data_folder = "data"

    K = load_camera_intrinsics(os.path.join(data_folder, "camera.dat"))

    frames = load_all_measurements(data_folder)

    vo = VisualOdometry(K)

    # Initialize with first two frames
    kpts0, desc0 = frames[0]
    kpts1, desc1 = frames[1]

    vo.process_first_two_frames(kpts0, desc0, kpts1, desc1)

    # Process remaining frames
    for kpts, desc in frames[2:]:
        vo.process_frame(kpts, desc)

    print("VO finished.")
    print(f"Total poses: {len(vo.poses)}")
    print(f"Total landmarks: {len(vo.landmarks)}")


if __name__ == "__main__":
    main()

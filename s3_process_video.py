import argparse
import os
import platform
import subprocess
import warnings
from os.path import join

import cv2
import imageio
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.python.solutions import hands, pose
from tqdm import tqdm

from helpers.utils import AnglesHelper
from helpers.visualization import Visualization

# system_name = platform.system()
system_name = "not_darwin"
pd.options.mode.copy_on_write = True
idx = pd.IndexSlice
warnings.filterwarnings("ignore")


def run_mediapipe(
    cap, frames, video_timestamps, sides, scales, hand_roi_size, process=False
):
    body_model_path = "models/mediapipe/pose_landmarker_lite.task"
    hands_model_path = "models/mediapipe/hand_landmarker.task"
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hands_model_path),
        num_hands=2,
        running_mode=VisionRunningMode.VIDEO,
    )
    hand_models = {
        "Right": mp.tasks.vision.HandLandmarker.create_from_options(options),
        "Left": mp.tasks.vision.HandLandmarker.create_from_options(options),
    }

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=body_model_path),
        running_mode=VisionRunningMode.VIDEO,
    )
    body_model = PoseLandmarker.create_from_options(options)

    hand_cols = hands.HandLandmark._member_names_ + [
        "BODY_WRIST",
        "ELBOW",
        "SHOULDER",
        "HIP",
    ]
    body_cols = pose.PoseLandmark._member_names_
    body_columns = pd.MultiIndex.from_product([["Body"], body_cols, ["x", "y", "z"]])
    right_hand_columns = pd.MultiIndex.from_product(
        [["Right"], hand_cols, ["x", "y", "z"]]
    )
    left_hand_columns = pd.MultiIndex.from_product(
        [["Left"], hand_cols, ["x", "y", "z"]]
    )

    columns = body_columns.append(right_hand_columns).append(left_hand_columns)
    joints_df = pd.DataFrame(index=frames, columns=columns)

    roi_half_size = int(hand_roi_size // 2)
    print("MediaPipe processing...")
    for frame_id in tqdm(frames):
        success, frame = cap.read()
        if not success:
            print(f"Finished MediaPipe processing at frame {frame_id}.")
            break
        if system_name == "Darwin":
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        body_results = body_model.detect_for_video(
            mp_image, video_timestamps[frame_id]
        ).pose_landmarks  # todo check
        if len(body_results) == 0:
            frame_id += 1
            continue

        for landmark_name in body_cols:
            joints_df.loc[frame_id, ("Body", landmark_name, "x")] = int(
                body_results[0][pose.PoseLandmark[landmark_name]].x * scales[0]
            )
            joints_df.loc[frame_id, ("Body", landmark_name, "y")] = int(
                body_results[0][pose.PoseLandmark[landmark_name]].y * scales[1]
            )
            joints_df.loc[frame_id, ("Body", landmark_name, "z")] = int(
                body_results[0][pose.PoseLandmark[landmark_name]].z * scales[2]
            )

        for side in sides:
            USE_ROI = False
            if not USE_ROI and not process:
                return None
            wrist = [
                joints_df.loc[frame_id, ("Body", f"{side.upper()}_WRIST", "x")],
                joints_df.loc[frame_id, ("Body", f"{side.upper()}_WRIST", "y")],
            ]
            if USE_ROI:
                x_start = max(0, wrist[0] - roi_half_size)
                x_end = min(scales[0], wrist[0] + roi_half_size)
                y_start = max(0, wrist[1] - roi_half_size)
                y_end = min(scales[1], wrist[1] + roi_half_size)
                cropped_frame = frame[y_start:y_end, x_start:x_end]
                rgb_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                if system_name != "Darwin":
                    rgb_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
                if frame_id == 1 and not process:
                    plt.imshow(rgb_cropped_frame)
                    plt.gca().set_xticks([0, roi_half_size * 2])
                    plt.gca().set_yticks([0, roi_half_size * 2])

                    plt.show()
                if frame_id == 2 and not process:
                    return None

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_cropped_frame
                )
            else:
                x_start = 0
                x_end = scales[0]
                y_start = 0
                y_end = scales[1]

            hands_results = (
                hand_models[side]
                .detect_for_video(mp_image, video_timestamps[frame_id])
                .hand_landmarks
            )
            if len(hands_results) == 0:
                continue
            elif len(hands_results) == 1:
                hand_id = 0

            else:
                x = joints_df.loc[frame_id, ("Body", f"{side.upper()}_WRIST", "x")]
                y = joints_df.loc[frame_id, ("Body", f"{side.upper()}_WRIST", "y")]
                target = np.array([x, y])

                candidates = np.zeros(2)
                x = x_start + hands_results[0][hands.HandLandmark["WRIST"]].x * (
                    x_end - x_start
                )
                y = y_start + hands_results[0][hands.HandLandmark["WRIST"]].y * (
                    y_end - y_start
                )
                candidates[0] = np.linalg.norm(target - np.array([x, y]))

                x = x_start + hands_results[1][hands.HandLandmark["WRIST"]].x * (
                    x_end - x_start
                )
                y = y_start + hands_results[1][hands.HandLandmark["WRIST"]].y * (
                    y_end - y_start
                )
                candidates[1] = np.linalg.norm(target - np.array([x, y]))
                hand_id = np.argmin(candidates)

            for landmark_name in hands.HandLandmark._member_names_:
                joints_df.loc[frame_id, (side, landmark_name, "x")] = (
                    x_start
                    + hands_results[hand_id][hands.HandLandmark[landmark_name]].x
                    * (x_end - x_start)
                )
                joints_df.loc[frame_id, (side, landmark_name, "y")] = (
                    y_start
                    + hands_results[hand_id][hands.HandLandmark[landmark_name]].y
                    * (y_end - y_start)
                )
                joints_df.loc[frame_id, (side, landmark_name, "z")] = (
                    x_start
                    + hands_results[hand_id][hands.HandLandmark[landmark_name]].z
                    * (x_end - x_start)
                )

        joints_df = joints_df.fillna(0)
    return joints_df


def update_left_right(joints_df):
    for key, value in {"Left": "LEFT", "Right": "RIGHT"}.items():
        for landmark_name in ["SHOULDER", "ELBOW", "HIP"]:
            joints_df.loc[:, idx[key, landmark_name, slice(None)]] = joints_df.loc[
                :, idx["Body", f"{value}_{landmark_name}", slice(None)]
            ].values

        # joints_df.loc[:, idx[key, 'BODY_WRIST', ['x', 'y']]] = joints_df.loc[:, idx[key, 'WRIST', ['x', 'y']]].values
        joints_df.loc[:, idx[key, "BODY_WRIST", ["x", "y"]]] = joints_df.loc[
            :, idx["Body", f"{value}_WRIST", ["x", "y"]]
        ].values

        joints_df.loc[:, idx[key, "BODY_WRIST", "z"]] = joints_df.loc[
            :, idx["Body", f"{value}_WRIST", "z"]
        ].values
    return joints_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture a video.")
    parser.add_argument("--data_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Experiment name"
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize the output")
    parser.add_argument(
        "--intact_hand", type=str, default=None, help="Intact hand (Right/Left)"
    )
    parser.add_argument("--hand_roi_size", type=int, default=800, help="Hand ROI size")
    parser.add_argument(
        "--plane_frames_start", type=int, default=0, help="Start of the plane frames"
    )
    parser.add_argument(
        "--plane_frames_end", type=int, default=20, help="End of the plane frames"
    )
    parser.add_argument("--video_start", type=int, default=0, help="Start of the video")
    parser.add_argument("--video_end", type=int, default=-1, help="End of the video")
    parser.add_argument("--process", action="store_true", help="Process the video")
    parser.add_argument(
        "--jorisThumb", action="store_true", help="Use Joris thumb angles"
    )
    args = parser.parse_args()

    sides = [args.intact_hand] if args.intact_hand else ["Right", "Left"]
    experiment_dir = join(args.data_dir, "experiments", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    input_video_path = join(args.data_dir, "video.mp4")

    vid = imageio.get_reader(input_video_path, "ffmpeg")
    vid_size = vid.get_meta_data()["size"]
    fps = vid.get_meta_data()["fps"]  # todo check
    vid.close()

    cap = cv2.VideoCapture(input_video_path)
    scales = (vid_size[0], vid_size[1], vid_size[0])
    n_frames = (
        cap.get(cv2.CAP_PROP_FRAME_COUNT) if args.video_end == -1 else args.video_end
    )
    frames = range(0, int(n_frames))
    video_timestamps = [int(frame * 1000 / fps) for frame in frames]
    joints_df = run_mediapipe(
        cap, frames, video_timestamps, sides, scales, args.hand_roi_size, args.process
    )
    cap.release()

    if not args.process:
        subprocess.run(["python", "video_gui.py", "--file", input_video_path])
        exit(0)

    joints_df = update_left_right(joints_df)
    joints_df.to_parquet(join(experiment_dir, "mediapipe_output.parquet"))
    joints_df = pd.read_parquet(join(experiment_dir, "mediapipe_output.parquet"))

    print("Correcting Z values...")
    for side in sides:
        # get upper arm length and forearm length:
        upper_arm_lengths = []
        forearm_lengths = []
        for i in range(args.plane_frames_start, args.plane_frames_end):
            shoulder = joints_df.loc[i, (side, "SHOULDER", ["x", "y"])].values
            elbow = joints_df.loc[i, (side, "ELBOW", ["x", "y"])].values
            wrist = joints_df.loc[i, (side, "WRIST", ["x", "y"])].values

            upper_arm_lengths.append(np.linalg.norm(shoulder - elbow))
            forearm_lengths.append(np.linalg.norm(elbow - wrist))
        average_upper_arm_length = np.mean(np.array(upper_arm_lengths))
        average_forearm_length = np.mean(np.array(forearm_lengths))

        upper_arm = (
            joints_df.loc[:, idx[side, "ELBOW", slice(None)]].values
            - joints_df.loc[:, idx[side, "SHOULDER", slice(None)]].values
        )
        forearm = (
            joints_df.loc[:, idx[side, "WRIST", slice(None)]].values
            - joints_df.loc[:, idx[side, "ELBOW", slice(None)]].values
        )
        upper_arm = upper_arm.astype(np.float64)
        forearm = forearm.astype(np.float64)

        missing_len_upper_arm = (
            average_upper_arm_length**2 - upper_arm[:, 0] ** 2 - upper_arm[:, 1] ** 2
        )
        missing_len_forearm = (
            average_forearm_length**2 - forearm[:, 0] ** 2 - forearm[:, 1] ** 2
        )

        missing_len_upper_arm = np.where(
            missing_len_upper_arm > 0, missing_len_upper_arm, 0
        )
        missing_len_forearm = np.where(missing_len_forearm > 0, missing_len_forearm, 0)

        upper_arm[:, 2] = np.sqrt(missing_len_upper_arm)
        forearm[:, 2] = np.sqrt(missing_len_forearm)

        joints_df.loc[:, idx["Body", f"{side.upper()}_ELBOW", "z"]] = (
            joints_df.loc[:, idx["Body", f"{side.upper()}_SHOULDER", "z"]].values
            + upper_arm[:, 2] * -1
        )
        joints_df.loc[:, idx["Body", f"{side.upper()}_WRIST", "z"]] = (
            joints_df.loc[:, idx["Body", f"{side.upper()}_ELBOW", "z"]].values
            + forearm[:, 2] * -1
        )

    joints_df = update_left_right(joints_df)
    joints_df.to_parquet(join(experiment_dir, "corrected.parquet"))

    corrected = pd.read_parquet(join(experiment_dir, "corrected.parquet"))
    anglesHelper = AnglesHelper()

    angles_df = anglesHelper.getArmAngles(corrected, sides)
    angles_df.to_parquet(join(experiment_dir, "angles.parquet"))

    smooth_angles_df = anglesHelper.apply_gaussian_smoothing(
        angles_df, sigma=1.5, radius=2
    )
    smooth_angles_df.to_parquet(join(experiment_dir, "smooth_angles.parquet"))

    start = args.video_start
    end = args.video_end if args.video_end != -1 else len(angles_df)

    angles_df = pd.read_parquet(join(experiment_dir, "angles.parquet"))
    angles_df = angles_df.loc[start:]
    angles_df.to_parquet(join(experiment_dir, "cropped_angles.parquet"))
    smooth_angles_df = pd.read_parquet(join(experiment_dir, "smooth_angles.parquet"))
    smooth_angles_df = smooth_angles_df.loc[start:]
    smooth_angles_df.to_parquet(join(experiment_dir, "cropped_smooth_angles.parquet"))
    emg = np.load(join(args.data_dir, "emg.npy"))

    ##### JUST FOR SHAUN - MULTIPLY CHANNEL 12 by 2
    # emg[:, 12] = emg[:, 12] * 2
    np.save(join(experiment_dir, "cropped_emg.npy"), emg[start:end])

    if args.visualize:
        df3d = pd.read_parquet(join(experiment_dir, "corrected.parquet"))
        if args.intact_hand is not None:
            df3d = AnglesHelper().mirror_pose(df3d, args.intact_hand)

        vis = Visualization(
            args.data_dir,
            args.experiment_name,
            df3d,
            start_frame=args.video_start,
            end_frame=end,
            name_addition="_corrected",
        )

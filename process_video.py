import argparse
from os.path import join

import cv2
import imageio

import mediapipe as mp
from mediapipe.python.solutions import pose
import numpy as np
from math import sqrt, ceil
import pandas as pd
pd.options.mode.copy_on_write = True
idx = pd.IndexSlice

from tqdm import tqdm
from helpers.utils import MergingHelper, AnglesHelper
from helpers.visualization import Visualization



def run_mediapipe(cap, frames, video_timestamps, intact_hand=None):
    body_model_path = 'models/mediapipe/pose_landmarker.task'
    hands_model_path = 'models/mediapipe/hand_landmarker.task'
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    mergingHelper = MergingHelper()

    options = mp.tasks.vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=hands_model_path)
                                                    , num_hands=2, running_mode=VisionRunningMode.VIDEO)
    hands_model = mp.tasks.vision.HandLandmarker.create_from_options(options)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=body_model_path),
        running_mode=VisionRunningMode.VIDEO)
    body_model = PoseLandmarker.create_from_options(options)

    body_cols = mergingHelper.map['Body'].keys()
    right_hand_cols = mergingHelper.map['Right'].keys()
    left_hand_cols = mergingHelper.map['Left'].keys()

    body_columns = pd.MultiIndex.from_product([['Body'], body_cols, ['x', 'y', 'z']])
    right_hand_columns = pd.MultiIndex.from_product([['Right'], right_hand_cols, ['x', 'y', 'z']])
    left_hand_columns = pd.MultiIndex.from_product([['Left'], left_hand_cols, ['x', 'y', 'z']])

    columns = body_columns.append(right_hand_columns).append(left_hand_columns)
    joints_df = pd.DataFrame(index=frames, columns=columns)

    frame_id = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Finished MediaPipe processing.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        body_results = body_model.detect_for_video(mp_image, int(1000 * video_timestamps[frame_id]))
        hands_results = hands_model.detect_for_video(mp_image, int(1000 * video_timestamps[frame_id]))
        pose_landmarks = mergingHelper.mergeLandmarks(hands_results, body_results, prior=intact_hand)

        coordinates = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]
        coordinates = np.asarray(coordinates)
        cols = {'Body': body_cols, 'Right': right_hand_cols, 'Left': left_hand_cols}
        for key, value in cols.items():
            pass
            for landmark_name in value:
                joints_df.loc[frame_id, (key, landmark_name, 'x')] = coordinates[
                mergingHelper.map[key][landmark_name], 0]
                joints_df.loc[frame_id, (key, landmark_name, 'y')] = coordinates[
                mergingHelper.map[key][landmark_name], 1]
                joints_df.loc[frame_id, (key, landmark_name, 'z')] = coordinates[
                mergingHelper.map[key][landmark_name], 2]


        frame_id += 1

    return joints_df


def update_left_right(joints_df):
    for key, value in {'Left': 'LEFT', 'Right': 'RIGHT'}.items():
        for landmark_name in ['SHOULDER', 'ELBOW', 'HIP']:
            joints_df.loc[:, idx[key, landmark_name, slice(None)]] = joints_df.loc[:, idx['Body', f'{value}_{landmark_name}', slice(None)]].values

        joints_df.loc[:, idx[key, 'BODY_WRIST', slice(None)]] = joints_df.loc[:,
                                                                   idx['Body', f'{value}_WRIST', slice(None)]].values
    return joints_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture a video.')
    parser.add_argument('--data_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--visualize', action='store_true', help='Visualize the output')
    parser.add_argument('--intact_hand', type=str, default=None, help='Intact hand')
    args = parser.parse_args()

    experiment_dir = join(args.data_dir, 'experiments', args.experiment_name)
    input_video_path = join(experiment_dir, "cropped_video.mp4")

    cap = cv2.VideoCapture(input_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Number of frames: {n_frames}")

    frames = range(0, int(n_frames))
    mediapipe_landmark_names = pose.PoseLandmark._member_names_
    video_timestamps = np.load(join(experiment_dir, "cropped_timestamps.npy"))
    joints_df = run_mediapipe(cap, frames, video_timestamps, args.intact_hand)

    vid = imageio.get_reader(input_video_path, 'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size']
    x_scale, y_scale = vid_size
    z_scale = x_scale

    idx = pd.IndexSlice
    joints_df.loc[:, idx[:, :, 'x']] = joints_df.loc[:, idx[:, :, 'x']] * x_scale
    joints_df.loc[:, idx[:, :, 'y']] = joints_df.loc[:, idx[:, :, 'y']] * y_scale
    joints_df.loc[:, idx[:, :, 'z']] = joints_df.loc[:, idx[:, :, 'z']] * z_scale
    cap.release()

    # joints_df = interpolate_missing_joints(joints_df, frames)
    joints_df.to_parquet(join(experiment_dir, "mediapipe_output.parquet"))

    joints_df = pd.read_parquet(join(experiment_dir, "mediapipe_output.parquet"))

    # get upper arm length and forearm length for the first 20 frames:
    upper_arm_lengths = []
    forearm_lengths = []
    sides = [args.intact_hand.upper()] if args.intact_hand else ['RIGHT', 'LEFT']
    for i in range(2):
        for side in sides:
            shoulder = joints_df.loc[i, ('Body', f'{side}_SHOULDER', ['x', 'y'])].values
            elbow = joints_df.loc[i, ('Body', f'{side}_ELBOW', ['x', 'y'])].values
            wrist = joints_df.loc[i, ('Body', f'{side}_WRIST', ['x', 'y'])].values

            upper_arm_lengths.append(np.linalg.norm(shoulder - elbow))
            forearm_lengths.append(np.linalg.norm(elbow - wrist))
    average_upper_arm_length = np.mean(np.array(upper_arm_lengths))
    average_forearm_length = np.mean(np.array(forearm_lengths))

    for i in range(len(joints_df.index)):
        # shoulder = joints_df.loc[i, idx['Body', 'RIGHT_SHOULDER', slice(None)]].values - joints_df.loc[i, idx['Body', 'LEFT_SHOULDER', slice(None)]].values
        # spine = joints_df.loc[i, idx['Body', 'HIPS', slice(None)]].values - joints_df.loc[i, idx['Body', 'CHEST', slice(None)]].values
        # body_normal = np.cross(shoulder, spine)
        # body_normal = body_normal / np.linalg.norm(body_normal)
        for side in ['RIGHT', 'LEFT']:
            upper_arm = joints_df.loc[i, idx['Body', f'{side}_ELBOW', slice(None)]].values - joints_df.loc[i, idx['Body', f'{side}_SHOULDER', slice(None)]].values
            forearm = joints_df.loc[i, idx['Body', f'{side}_WRIST', slice(None)]].values - joints_df.loc[i, idx['Body', f'{side}_ELBOW', slice(None)]].values

            # upper_arm_z_sign = np.sign(np.dot(upper_arm, body_normal))
            # forearm_z_sign = np.sign(np.dot(upper_arm, body_normal))
            upper_arm_z_sign = -1
            forearm_z_sign = -1

            missing_len = average_upper_arm_length**2 - upper_arm[0]**2 - upper_arm[1]**2
            if missing_len > 0:
                upper_arm[2] = sqrt(missing_len)
            else:
                upper_arm[2] = 0

            missing_len = average_forearm_length ** 2 - forearm[0] ** 2 - forearm[1] ** 2
            if missing_len > 0:
                forearm[2] = sqrt(missing_len)
            else:
                forearm[2] = 0

            joints_df.loc[i, idx['Body', f'{side}_ELBOW', 'z']] = joints_df.loc[i, idx['Body', f'{side}_SHOULDER', 'z']] + upper_arm[2] * upper_arm_z_sign
            joints_df.loc[i, idx['Body', f'{side}_WRIST', 'z']] = joints_df.loc[i, idx['Body', f'{side}_ELBOW', 'z']] + forearm[2] * forearm_z_sign
            # todo update LR
            # todo scaling depending on elbow z value (relative to shoulder / plane)
            pass
    joints_df = update_left_right(joints_df)
    joints_df.to_parquet(join(experiment_dir, "corrected.parquet"))


    if args.visualize:
        df2d = pd.read_parquet(join(experiment_dir, "mediapipe_output.parquet"))

        df3d = pd.read_parquet(join(experiment_dir, "mediapipe_output.parquet"))
        vis = Visualization(experiment_dir, df2d, df3d, alternative=False)

        df3d = pd.read_parquet(join(experiment_dir, "corrected.parquet"))
        vis = Visualization(experiment_dir, df2d, df3d, name_addition="_corrected")



    # corrected = pd.read_parquet(join(experiment_dir, "corrected.parquet"))
    # anglesHelper = AnglesHelper()
    # angles_df = anglesHelper.getArmAngles(corrected)
    # angles_df.to_parquet(join(experiment_dir, "angles.parquet"))

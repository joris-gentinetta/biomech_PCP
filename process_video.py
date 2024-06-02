import argparse
from os.path import join

import cv2
import imageio
import time
import subprocess

import mediapipe as mp
from mediapipe.python.solutions import pose, hands
import numpy as np
from math import sqrt, ceil
import pandas as pd
pd.options.mode.copy_on_write = True
idx = pd.IndexSlice

from tqdm import tqdm
from helpers.utils import AnglesHelper
from helpers.visualization import Visualization

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def run_mediapipe(cap, frames, video_timestamps, sides, scales, hand_roi_size, process=False):
    body_model_path = 'models/mediapipe/pose_landmarker_heavy.task'
    hands_model_path = 'models/mediapipe/hand_landmarker.task'
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = mp.tasks.vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=hands_model_path)
                                                    , num_hands=2, running_mode=VisionRunningMode.VIDEO)
    hand_models = {'Right': mp.tasks.vision.HandLandmarker.create_from_options(options), 'Left': mp.tasks.vision.HandLandmarker.create_from_options(options)}

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=body_model_path),
        running_mode=VisionRunningMode.VIDEO)
    body_model = PoseLandmarker.create_from_options(options)

    hand_cols = hands.HandLandmark._member_names_ + ['BODY_WRIST', 'ELBOW', 'SHOULDER', 'HIP']
    body_cols = pose.PoseLandmark._member_names_
    body_columns = pd.MultiIndex.from_product([['Body'], body_cols, ['x', 'y', 'z']])
    right_hand_columns = pd.MultiIndex.from_product([['Right'], hand_cols, ['x', 'y', 'z']])
    left_hand_columns = pd.MultiIndex.from_product([['Left'], hand_cols, ['x', 'y', 'z']])

    columns = body_columns.append(right_hand_columns).append(left_hand_columns)
    joints_df = pd.DataFrame(index=frames, columns=columns)

    roi_half_size = int(hand_roi_size // 2)
    print("MediaPipe processing...")
    for frame_id in tqdm(frames):
        success, frame = cap.read()
        if not success:
            print(f"Finished MediaPipe processing at frame {frame_id}.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        body_results = body_model.detect_for_video(mp_image, int(1000 * video_timestamps[frame_id])).pose_landmarks
        if len(body_results) == 0:
            frame_id += 1
            continue

        for landmark_name in body_cols:
            joints_df.loc[frame_id, ('Body', landmark_name, 'x')] = int(body_results[0][pose.PoseLandmark[landmark_name]].x * scales[0])
            joints_df.loc[frame_id, ('Body', landmark_name, 'y')] = int(body_results[0][pose.PoseLandmark[landmark_name]].y * scales[1])
            joints_df.loc[frame_id, ('Body', landmark_name, 'z')] = int(body_results[0][pose.PoseLandmark[landmark_name]].z * scales[2])

        for side in sides:
            wrist = [joints_df.loc[frame_id, ('Body', f'{side.upper()}_WRIST', 'x')], joints_df.loc[frame_id, ('Body', f'{side.upper()}_WRIST', 'y')]]
            x_start = max(0, wrist[0] - roi_half_size)
            x_end = min(scales[0], wrist[0] + roi_half_size)
            y_start = max(0, wrist[1] - roi_half_size)
            y_end = min(scales[1], wrist[1] + roi_half_size)
            cropped_frame = frame[y_start:y_end, x_start:x_end]
            #todo make sure the frame is not empty
            rgb_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            if frame_id == 1:
                plt.imshow(rgb_cropped_frame)
                # add axis values:
                plt.gca().set_xticks([0, roi_half_size * 2])
                plt.gca().set_yticks([0, roi_half_size * 2])

                plt.show()
            if frame_id == 2 and not process:
                return None

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_cropped_frame)

            hands_results = hand_models[side].detect_for_video(mp_image, int(1000 * video_timestamps[frame_id])).hand_landmarks
            if len(hands_results) == 0:
                continue
            elif len(hands_results) == 1:
                hand_id = 0

            else:
                x = joints_df.loc[frame_id, ('Body', f'{side.upper()}_WRIST', 'x')]
                y = joints_df.loc[frame_id, ('Body', f'{side.upper()}_WRIST', 'y')]
                target = np.array([x, y])

                candidates = np.zeros(2)
                x = x_start + hands_results[0][hands.HandLandmark['WRIST']].x * (x_end - x_start)
                y = y_start + hands_results[0][hands.HandLandmark['WRIST']].y * (y_end - y_start)
                candidates[0] = np.linalg.norm(target - np.array([x, y]))

                x = x_start + hands_results[1][hands.HandLandmark['WRIST']].x * (x_end - x_start)
                y = y_start + hands_results[1][hands.HandLandmark['WRIST']].y * (y_end - y_start)
                candidates[1] = np.linalg.norm(target - np.array([x, y]))
                hand_id = np.argmin(candidates)

            for landmark_name in hands.HandLandmark._member_names_:
                joints_df.loc[frame_id, (side, landmark_name, 'x')] = x_start + hands_results[hand_id][hands.HandLandmark[landmark_name]].x * (x_end - x_start)
                joints_df.loc[frame_id, (side, landmark_name, 'y')] = y_start + hands_results[hand_id][hands.HandLandmark[landmark_name]].y * (y_end - y_start)
                joints_df.loc[frame_id, (side, landmark_name, 'z')] = x_start + hands_results[hand_id][hands.HandLandmark[landmark_name]].z * (x_end - x_start)

        joints_df = joints_df.fillna(0)
    return joints_df


def update_left_right(joints_df):
    for key, value in {'Left': 'LEFT', 'Right': 'RIGHT'}.items():
        for landmark_name in ['SHOULDER', 'ELBOW', 'HIP']:
            joints_df.loc[:, idx[key, landmark_name, slice(None)]] = joints_df.loc[:, idx['Body', f'{value}_{landmark_name}', slice(None)]].values

        joints_df.loc[:, idx[key, 'BODY_WRIST', ['x', 'y']]] = joints_df.loc[:, idx[key, 'WRIST', ['x', 'y']]].values
        joints_df.loc[:, idx[key, 'BODY_WRIST', 'z']] = joints_df.loc[:, idx['Body', f'{value}_WRIST', 'z']].values
    return joints_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture a video.')
    parser.add_argument('--data_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--visualize', action='store_true', help='Visualize the output')
    parser.add_argument('--intact_hand', type=str, default=None, help='Intact hand')
    parser.add_argument('--hand_roi_size', type=int, default=800, help='Hand ROI size')
    parser.add_argument('--plane_frames_start', type=int, default=0, help='Start of the plane frames')
    parser.add_argument('--plane_frames_end', type=int, default=20, help='End of the plane frames')
    parser.add_argument('--process', action='store_true', help='Process the video')
    args = parser.parse_args()

    start = time.time()
    sides = [args.intact_hand] if args.intact_hand else ['Right', 'Left']
    experiment_dir = join(args.data_dir, 'experiments', args.experiment_name)
    input_video_path = join(experiment_dir, "cropped_video.mp4")


    cap = cv2.VideoCapture(input_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    vid = imageio.get_reader(input_video_path, 'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size']
    scales = (vid_size[0], vid_size[1], vid_size[0])
    frames = range(0, int(n_frames))
    mediapipe_landmark_names = pose.PoseLandmark._member_names_

    video_timestamps = np.load(join(experiment_dir, "cropped_video_timestamps.npy"))
    joints_df = run_mediapipe(cap, frames, video_timestamps, sides, scales, args.hand_roi_size, args.process)
    cap.release()

    if not args.process:
        subprocess.run(['python', 'video_gui.py', '--file', input_video_path])
        exit(0)

    joints_df = update_left_right(joints_df)
    joints_df.to_parquet(join(experiment_dir, "mediapipe_output.parquet"))

    print('Correcting Z values...')
    for side in sides:
        # get upper arm length and forearm length:
        upper_arm_lengths = []
        forearm_lengths = []
        for i in range(args.plane_frames_start, args.plane_frames_end):
            shoulder = joints_df.loc[i, (side, 'SHOULDER', ['x', 'y'])].values
            elbow = joints_df.loc[i, (side, 'ELBOW', ['x', 'y'])].values
            wrist = joints_df.loc[i, (side, 'WRIST', ['x', 'y'])].values

            upper_arm_lengths.append(np.linalg.norm(shoulder - elbow))
            forearm_lengths.append(np.linalg.norm(elbow - wrist))
        average_upper_arm_length = np.mean(np.array(upper_arm_lengths))
        average_forearm_length = np.mean(np.array(forearm_lengths))


        # Calculate upper_arm and forearm for all rows at once
        upper_arm = joints_df.loc[:, idx[side, 'ELBOW', slice(None)]].values - joints_df.loc[:,
                                                                                idx[side, 'SHOULDER', slice(None)]].values
        forearm = joints_df.loc[:, idx[side, 'WRIST', slice(None)]].values - joints_df.loc[:,
                                                                              idx[side, 'ELBOW', slice(None)]].values
        upper_arm = upper_arm.astype(np.float64)
        forearm = forearm.astype(np.float64)

        # Calculate missing_len for upper_arm and forearm for all rows at once
        missing_len_upper_arm = average_upper_arm_length ** 2 - upper_arm[:, 0] ** 2 - upper_arm[:, 1] ** 2
        missing_len_forearm = average_forearm_length ** 2 - forearm[:, 0] ** 2 - forearm[:, 0] ** 2

        missing_len_upper_arm = np.where(missing_len_upper_arm > 0, missing_len_upper_arm, 0)
        missing_len_forearm = np.where(missing_len_forearm > 0, missing_len_forearm, 0)

        upper_arm[:, 2] = np.sqrt(missing_len_upper_arm)
        forearm[:, 2] = np.sqrt(missing_len_forearm)

        joints_df.loc[:, idx['Body', f'{side.upper()}_ELBOW', 'z']] = joints_df.loc[:, idx[
                                                                                           'Body', f'{side.upper()}_SHOULDER', 'z']].values + upper_arm[ :, 2] * -1
        joints_df.loc[:, idx['Body', f'{side.upper()}_WRIST', 'z']] = joints_df.loc[:, idx[
                                                                                           'Body', f'{side.upper()}_ELBOW', 'z']].values + forearm[:, 2] * -1

    joints_df = update_left_right(joints_df)
    joints_df.to_parquet(join(experiment_dir, "corrected.parquet"))
    print(f"Time taken: {(time.time() - start)/60} minutes.")

    if args.visualize:
        df3d = pd.read_parquet(join(experiment_dir, "corrected.parquet"))
        vis = Visualization(experiment_dir, df3d, name_addition="_corrected")

    corrected = pd.read_parquet(join(experiment_dir, "corrected.parquet"))
    anglesHelper = AnglesHelper()
    angles_df = anglesHelper.getArmAngles(corrected, sides)
    angles_df.to_parquet(join(experiment_dir, "angles.parquet"))

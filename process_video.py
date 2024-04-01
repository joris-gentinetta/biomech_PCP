import argparse
from os.path import join

import cv2
import imageio
import mediapipe as mp
import numpy as np
import pandas as pd
pd.options.mode.copy_on_write = True
idx = pd.IndexSlice

from mediapipe.python.solutions import pose
from tqdm import tqdm

from helpers.MotionBert import MotionBert, MOTIONBERT_MAP
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
            print("Finished processing video.")
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


def interpolate_missing_joints(joints_df, frames):
    new_columns = pd.MultiIndex.from_product([['Body'], ['HIPS', 'CHEST', 'SPINE', 'JAW'], ['x', 'y', 'z']])
    joints_df = pd.concat([joints_df, pd.DataFrame(index=frames, columns=new_columns)], axis=1)
    for frame_id in frames:
        # use hand wrist instead of body wrist:
        joints_df.loc[frame_id, idx["Body", "LEFT_WRIST", slice(None)]] = joints_df.loc[
            idx[frame_id, ('Left', 'WRIST', slice(None))]].values
        joints_df.loc[frame_id, idx["Body", "RIGHT_WRIST", slice(None)]] = joints_df.loc[
            idx[frame_id, ('Right', 'WRIST', slice(None))]].values

        # interpolate missing joints:
        joints_df.loc[frame_id, idx["Body", "HIPS", slice(None)]] = (joints_df.loc[idx[
            frame_id, ('Body', 'LEFT_HIP', slice(None))]].values +
                                                                 joints_df.loc[idx[frame_id, (
                                                                 'Body', 'RIGHT_HIP', slice(None))]].values) / 2
        joints_df.loc[frame_id, idx["Body", "CHEST", slice(None)]] = (joints_df.loc[idx[
            frame_id, ('Body', 'LEFT_SHOULDER', slice(None))]].values +
                                                                  joints_df.loc[idx[frame_id, (
                                                                  'Body', 'RIGHT_SHOULDER', slice(None))]].values) / 2
        joints_df.loc[frame_id, idx["Body", "SPINE", slice(None)]] = (joints_df.loc[
                                                                      idx[frame_id, ('Body', "CHEST", slice(None))]].values +
                                                                  joints_df.loc[
                                                                      idx[frame_id, ('Body', "HIPS", slice(None))]].values) / 2
        joints_df.loc[frame_id, idx["Body", "JAW", slice(None)]] = (joints_df.loc[idx[
            frame_id, ('Body', 'NOSE', slice(None))]].values +
                                                                joints_df.loc[
                                                                    idx[frame_id, ('Body', "CHEST", slice(None))]].values) / 2
    return joints_df


def get_motionbert_input(joints_df, probing_point, max_len):
    # TODO JAW SHOULD BE NOSE, NOSE SHOULD BE HEAD
    n_frames = len(joints_df.index)
    motionbert_input = np.zeros((n_frames, 17, 2))
    idx = pd.IndexSlice
    for i in range(n_frames):
        for j, landmark_name in enumerate(MOTIONBERT_MAP):
            motionbert_input[i, j, 0] = joints_df.loc[i, idx['Body', landmark_name, 'x']]
            motionbert_input[i, j, 1] = joints_df.loc[i, idx['Body', landmark_name, 'y']]

    first_row = motionbert_input[0]
    prepended_rows = np.repeat(first_row[np.newaxis, :], probing_point, axis=0)

    last_row = motionbert_input[-1]
    postpended_rows = np.repeat(last_row[np.newaxis, :], max_len, axis=0)

    motionbert_input = np.concatenate((prepended_rows, motionbert_input, postpended_rows), axis=0)

    return motionbert_input

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

    joints_df = interpolate_missing_joints(joints_df, frames)
    joints_df.to_parquet(join(experiment_dir, "mediapipe_output.parquet"))

    mb = MotionBert()
    motionbert_input = get_motionbert_input(joints_df, mb.probing_point, mb.model.maxlen)

    idx = pd.IndexSlice
    for i in tqdm(range(len(joints_df.index))):
        keypoints = mb.get_3d_keypoints(
            motionbert_input[i:i + mb.model.maxlen], vid_size)
        for lm_i, landmark_name in enumerate(MOTIONBERT_MAP):
            joints_df.loc[i, idx[ 'Body', landmark_name, slice(None)]] = keypoints[lm_i]

    mergingHelper = MergingHelper()
    body_cols = MOTIONBERT_MAP
    side_cols = mergingHelper.map['Right'].keys()

    body_columns = pd.MultiIndex.from_product([['Body'], body_cols, ['x', 'y', 'z']])
    right_side_columns = pd.MultiIndex.from_product([['Right'], side_cols, ['x', 'y', 'z']])
    left_side_columns = pd.MultiIndex.from_product([['Left'], side_cols, ['x', 'y', 'z']])

    joints_df = update_left_right(joints_df)

    joints_df.to_parquet(join(experiment_dir, "motionbert_output.parquet"))

    if args.visualize:
        df2d = pd.read_parquet(join(experiment_dir, "mediapipe_output.parquet"))
        df3d = pd.read_parquet(join(experiment_dir, "motionbert_output.parquet"))
        vis = Visualization(experiment_dir, df2d, df3d)

    motionbert_output = pd.read_parquet(join(experiment_dir, "motionbert_output.parquet"))
    anglesHelper = AnglesHelper()
    angles_df = anglesHelper.getArmAngles(motionbert_output)
    angles_df.to_parquet(join(experiment_dir, "angles.parquet"))

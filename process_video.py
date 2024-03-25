import argparse
from os.path import join

import cv2
import imageio
import mediapipe as mp
import numpy as np
import pandas as pd
pd.options.mode.copy_on_write = True
from mediapipe.python.solutions import pose
from tqdm import tqdm

from helpers.MotionBert import MotionBert, MOTIONBERT_MAP
from helpers.utils import MergingHelper, AnglesHelper
from helpers.visualization import Visualization



def get_mediapipe_df(cap, frames, video_timestamps, intact_hand=None):
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
    mediapipe_df = pd.DataFrame(index=frames, columns=columns)

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
        for landmark_name in body_cols:
            mediapipe_df.loc[frame_id, ('Body', landmark_name, 'x')] = coordinates[
                mergingHelper.map['Body'][landmark_name], 0]
            mediapipe_df.loc[frame_id, ('Body', landmark_name, 'y')] = coordinates[
                mergingHelper.map['Body'][landmark_name], 1]
            mediapipe_df.loc[frame_id, ('Body', landmark_name, 'z')] = coordinates[
                mergingHelper.map['Body'][landmark_name], 2]

        for landmark_name in right_hand_cols:
            mediapipe_df.loc[frame_id, ('Right', landmark_name, 'x')] = coordinates[
                mergingHelper.map['Right'][landmark_name], 0]
            mediapipe_df.loc[frame_id, ('Right', landmark_name, 'y')] = coordinates[
                mergingHelper.map['Right'][landmark_name], 1]
            mediapipe_df.loc[frame_id, ('Right', landmark_name, 'z')] = coordinates[
                mergingHelper.map['Right'][landmark_name], 2]

        for landmark_name in left_hand_cols:
            mediapipe_df.loc[frame_id, ('Left', landmark_name, 'x')] = coordinates[
                mergingHelper.map['Left'][landmark_name], 0]
            mediapipe_df.loc[frame_id, ('Left', landmark_name, 'y')] = coordinates[
                mergingHelper.map['Left'][landmark_name], 1]
            mediapipe_df.loc[frame_id, ('Left', landmark_name, 'z')] = coordinates[
                mergingHelper.map['Left'][landmark_name], 2]

        frame_id += 1

    return mediapipe_df


def get_motionbert_df(mediapipe_df, frames):
    motionbert_columns = pd.MultiIndex.from_product([MOTIONBERT_MAP, ['x', 'y', 'z']],
                                                    names=["landmark_name", "coordinate"])
    motionbert_df = pd.DataFrame(index=frames, columns=motionbert_columns)
    body_cols = mediapipe_df.columns.get_level_values(1)[mediapipe_df.columns.get_level_values(0) == 'Body'].unique()
    idx = pd.IndexSlice
    for frame_id in frames:
        for landmark_name in MOTIONBERT_MAP:
            if landmark_name in body_cols:
                motionbert_df.loc[frame_id, idx[landmark_name, slice(None)]] = mediapipe_df.loc[
                    idx[frame_id, ('Body', landmark_name, slice(None))]].values

        motionbert_df.loc[frame_id, idx["LEFT_WRIST", slice(None)]] = mediapipe_df.loc[
            idx[frame_id, ('Left', 'WRIST', slice(None))]].values
        motionbert_df.loc[frame_id, idx["RIGHT_WRIST", slice(None)]] = mediapipe_df.loc[
            idx[frame_id, ('Right', 'WRIST', slice(None))]].values

        motionbert_df.loc[frame_id, idx["HIPS", slice(None)]] = (mediapipe_df.loc[idx[
            frame_id, ('Body', 'LEFT_HIP', slice(None))]].values +
                                                                 mediapipe_df.loc[idx[frame_id, (
                                                                 'Body', 'RIGHT_HIP', slice(None))]].values) / 2
        motionbert_df.loc[frame_id, idx["CHEST", slice(None)]] = (mediapipe_df.loc[idx[
            frame_id, ('Body', 'LEFT_SHOULDER', slice(None))]].values +
                                                                  mediapipe_df.loc[idx[frame_id, (
                                                                  'Body', 'RIGHT_SHOULDER', slice(None))]].values) / 2
        motionbert_df.loc[frame_id, idx["SPINE", slice(None)]] = (motionbert_df.loc[
                                                                      idx[frame_id, ("CHEST", slice(None))]].values +
                                                                  motionbert_df.loc[
                                                                      idx[frame_id, ("HIPS", slice(None))]].values) / 2
        motionbert_df.loc[frame_id, idx["JAW", slice(None)]] = (mediapipe_df.loc[idx[
            frame_id, ('Body', 'NOSE', slice(None))]].values +
                                                                motionbert_df.loc[
                                                                    idx[frame_id, ("CHEST", slice(None))]].values) / 2
    return motionbert_df


def get_motionbert_input(motionbert_df, probing_point, max_len):
    # TODO JAW SHOULD BE NOSE, NOSE SHOULD BE HEAD
    n_frames = len(motionbert_df.index)
    motionbert_input = np.zeros((n_frames, 17, 2))
    idx = pd.IndexSlice
    for i in range(n_frames):
        for j, landmark_name in enumerate(MOTIONBERT_MAP):
            motionbert_input[i, j, 0] = motionbert_df.loc[i, idx[landmark_name, 'x']]
            motionbert_input[i, j, 1] = motionbert_df.loc[i, idx[landmark_name, 'y']]

    first_row = motionbert_input[0]
    prepended_rows = np.repeat(first_row[np.newaxis, :], probing_point, axis=0)

    last_row = motionbert_input[-1]
    postpended_rows = np.repeat(last_row[np.newaxis, :], max_len, axis=0)

    motionbert_input = np.concatenate((prepended_rows, motionbert_input, postpended_rows), axis=0)

    return motionbert_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture a video.')
    parser.add_argument('--data_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize the output')
    parser.add_argument('--intact_hand', type=str, default=None, help='Intact hand')
    args = parser.parse_args()

    input_video_path = join(args.data_dir, "cropped_video.mp4")

    cap = cv2.VideoCapture(input_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Number of frames: {n_frames}")

    frames = range(0, int(n_frames))
    mediapipe_landmark_names = pose.PoseLandmark._member_names_
    video_timestamps = np.load(join(args.data_dir, "cropped_timestamps.npy"))
    mediapipe_df = get_mediapipe_df(cap, frames, video_timestamps, args.intact_hand)

    vid = imageio.get_reader(input_video_path, 'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size']
    x_scale, y_scale = vid_size  # todo: check if this is correct
    z_scale = x_scale

    idx = pd.IndexSlice
    mediapipe_df.loc[:, idx[:, :, 'x']] = mediapipe_df.loc[:, idx[:, :, 'x']] * x_scale
    mediapipe_df.loc[:, idx[:, :, 'y']] = mediapipe_df.loc[:, idx[:, :, 'y']] * y_scale
    mediapipe_df.loc[:, idx[:, :, 'z']] = mediapipe_df.loc[:, idx[:, :, 'z']] * z_scale
    cap.release()

    motionbert_df = get_motionbert_df(mediapipe_df, frames)
    motionbert_df.to_parquet(join(args.data_dir, "motionbert_input.parquet"))

    mb = MotionBert()
    motionbert_input = get_motionbert_input(motionbert_df, mb.probing_point, mb.model.maxlen)

    idx = pd.IndexSlice
    for i in tqdm(range(len(motionbert_df.index))):
        keypoints = mb.get_3d_keypoints(
            motionbert_input[i:i + mb.model.maxlen], vid_size)
        for lm_i, landmark_name in enumerate(MOTIONBERT_MAP):
            motionbert_df.loc[i, idx[landmark_name, slice(None)]] = keypoints[lm_i]

    motionbert_df.to_parquet(join(args.data_dir, "motionbert_output.parquet"))
    mergingHelper = MergingHelper()
    body_cols = MOTIONBERT_MAP
    side_cols = mergingHelper.map['Right'].keys()

    body_columns = pd.MultiIndex.from_product([['Body'], body_cols, ['x', 'y', 'z']])
    right_side_columns = pd.MultiIndex.from_product([['Right'], side_cols, ['x', 'y', 'z']])
    left_side_columns = pd.MultiIndex.from_product([['Left'], side_cols, ['x', 'y', 'z']])

    columns = body_columns.append(right_side_columns).append(left_side_columns)
    output_df = pd.DataFrame(index=frames, columns=columns)

    for landmark_name in MOTIONBERT_MAP:
        output_df.loc[:, idx['Body', landmark_name, slice(None)]] = motionbert_df.loc[:, idx[landmark_name, slice(None)]].values

    output_df.loc[:, idx['Left', 'SHOULDER', slice(None)]] = output_df.loc[:, idx['Body', 'LEFT_SHOULDER', slice(None)]].values
    output_df.loc[:, idx['Left', 'ELBOW', slice(None)]] = output_df.loc[:, idx['Body', 'LEFT_ELBOW', slice(None)]].values
    output_df.loc[:, idx['Left', 'HIP', slice(None)]] = output_df.loc[:, idx['Body', 'LEFT_HIP', slice(None)]].values
    output_df.loc[:, idx['Left', 'BODY_WRIST', slice(None)]] = output_df.loc[:,
                                                               idx['Body', 'LEFT_WRIST', slice(None)]].values

    output_df.loc[:, idx['Right', 'SHOULDER', slice(None)]] = output_df.loc[:,
                                                              idx['Body', 'RIGHT_SHOULDER', slice(None)]].values
    output_df.loc[:, idx['Right', 'ELBOW', slice(None)]] = output_df.loc[:, idx['Body', 'RIGHT_ELBOW', slice(None)]].values
    output_df.loc[:, idx['Right', 'HIP', slice(None)]] = output_df.loc[:, idx['Body', 'RIGHT_HIP', slice(None)]].values
    output_df.loc[:, idx['Right', 'BODY_WRIST', slice(None)]] = output_df.loc[:,
                                                                idx['Body', 'RIGHT_WRIST', slice(None)]].values

    for col in mergingHelper.hand_keys:
        output_df.loc[:, idx['Left', col, slice(None)]] = mediapipe_df.loc[:, idx['Left', col, slice(None)]].values
        output_df.loc[:, idx['Right', col, slice(None)]] = mediapipe_df.loc[:, idx['Right', col, slice(None)]].values

    output_df.to_parquet(join(args.data_dir, "output.parquet"))

    if args.visualize:
        df2d = pd.read_parquet(join(args.data_dir, "motionbert_input.parquet"))
        df3d = pd.read_parquet(join(args.data_dir, "motionbert_output.parquet"))
        vis = Visualization(args.data_dir, df2d, df3d)

    output_df = pd.read_parquet(join(args.data_dir, "output.parquet"))
    anglesHelper = AnglesHelper()
    angles_df = anglesHelper.getArmAngles(output_df)
    angles_df.to_parquet(join(args.data_dir, "angles.parquet"))

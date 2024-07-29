
import argparse
from os.path import join

import cv2
import imageio
import subprocess

import mediapipe as mp
from mediapipe.python.solutions import pose, hands
import numpy as np
import pandas as pd
pd.options.mode.copy_on_write = True
idx = pd.IndexSlice

from tqdm import tqdm
from helpers.utils import AnglesHelper
from helpers.EMGClass import EMG

from helpers.visualization import Visualization

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from time import time, sleep
import wandb
from tqdm import tqdm
import argparse
import math
import os
import yaml
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import numpy as np
from os.path import join
import wandb
import multiprocessing
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from helpers.models import TimeSeriesRegressorWrapper
from tqdm import tqdm
from os.path import join
torch.autograd.set_detect_anomaly(True)
from queue import Queue

from helpers.predict_utils import Config, get_data, train_model


from helpers.predict_utils import Config, train_model, OLDataset, TSDataLoader, evaluate_model, EarlyStopper
import cv2


from threading import Thread
from queue import Queue
class WebcamVideoStream:
    def __init__(self, src=0, queueSize=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)
        self.emg = EMG()
        self.emg.startCommunication()
        self.emg_timestep = np.asarray(self.emg.normedEMG)


    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, frame) = self.stream.read()
            self.emg_timestep = np.asarray(self.emg.normedEMG)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            if not self.Q.full():
                self.Q.put((self.frame, self.emg_timestep))


    def read(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True
        self.stream.release()  # Release the video stream resource
        while not self.Q.empty():  # Clear the queue
            self.Q.get()


# from multiprocessing import Process, Queue
# class WebcamVideoStream:
#     def __init__(self, src=0, queueSize=0):
#         self.src = src
#         self.stopped = False
#         self.Q = Queue(maxsize=queueSize)
#
#     def start(self):
#         self.process = Process(target=self.update, args=())
#         self.process.daemon = True
#         self.process.start()
#         return self
#
#     def update(self):
#         self.stream = cv2.VideoCapture(self.src)
#         while True:
#             if self.stopped:
#                 self.stream.release()
#                 return
#             (grabbed, frame) = self.stream.read()
#             if not self.Q.full():
#                 self.Q.put(frame)
#
#     def read(self):
#         return self.Q.get()
#
#     def stop(self):
#         self.stopped = True
#         self.process.join()  # Ensure the process finishes
#         while not self.Q.empty():  # Clear the queue
#             self.Q.get()



def update_left_right(joints_df):
    for key, value in {'Left': 'LEFT', 'Right': 'RIGHT'}.items():
        for landmark_name in ['SHOULDER', 'ELBOW', 'HIP']:
            joints_df.loc[:, idx[key, landmark_name, slice(None)]] = joints_df.loc[:, idx
                                                                                          ['Body', f'{value}_{landmark_name}', slice
                                                                                              (None)]].values

        joints_df.loc[:, idx[key, 'BODY_WRIST', ['x', 'y']]] = joints_df.loc[:, idx[key, 'WRIST', ['x', 'y']]].values
        joints_df.loc[:, idx[key, 'BODY_WRIST', 'z']] = joints_df.loc[:, idx['Body', f'{value}_WRIST', 'z']].values
    return joints_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Timeseries data analysis')
    parser.add_argument('--person_dir', type=str, required=True, help='Person directory')
    parser.add_argument('--intact_hand', type=str, required=True, help='Intact hand (Right/Left)')
    parser.add_argument('--config_name', type=str, required=True, help='Training configuration')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--allow_tf32', action='store_true', help='Allow TF32')
    parser.add_argument('-v', '--visualize', action='store_true', help='Plot data exploration results')
    parser.add_argument('-hs', '--hyperparameter_search', action='store_true', help='Perform hyperparameter search')
    parser.add_argument('-t', '--test', action='store_true', help='Test the model')
    parser.add_argument('-s', '--save_model', action='store_true', help='Save a model')
    parser.add_argument('--offline', action='store_true', help='Offline training')
    args = parser.parse_args()

    epoch_len = 1000
    sampling_frequency = 60
    calibration_frames = 60 # todo

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Using CUDA')

        # List available GPUs
        if args.multi_gpu:
            n_gpus = torch.cuda.device_count()
            print(f'Number of available GPUs: {n_gpus}')
            for i in range(n_gpus):
                print(f'GPU{i}: {torch.cuda.get_device_name(i)}')

    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print('Using MPS')
    else:
        device = torch.device("cpu")
        print('Using CPU')

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        print('TF32 enabled')

    with open(join('data', args.person_dir, 'configs', f'{args.config_name}.yaml'), 'r') as file:
        wandb_config = yaml.safe_load(file)
        config = Config(wandb_config)

    emg_channels = [int(feature[1]) for feature in config.features]


    data_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') for recording in
                 config.recordings]

    test_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') for recording in
                 config.test_recordings] if config.test_recordings is not None else None

    trainsets, testsets, combined_sets = get_data(config, data_dirs, args.intact_hand, visualize=args.visualize,
                                                  test_dirs=test_dirs)




    with open(join('data', args.person_dir, 'configs', f'{args.config_name}.yaml'), 'r') as file:
        wandb_config = yaml.safe_load(file)
        config = Config(wandb_config)

    if args.offline:
        with wandb.init(mode=config.wandb_mode, project=config.wandb_project, name=config.name, config=config):
            config = wandb.config

            model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features), output_size=len(config.targets),
                                               **config)
            model.to(device)
            model.train()
            # if config.model_type == 'ActivationAndBiophys':  # todo
            #     for param in model.model.biophys_model.parameters():
            #         param.requires_grad = False if epoch < config.biophys_config['n_freeze_epochs'] else True

            train_dataset = OLDataset(trainsets, config.features, config.targets, device=device)
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False)
            best_val_loss = float('inf')
            early_stopper = EarlyStopper(patience=config.early_stopping_patience, min_delta=config.early_stopping_delta)
            print('Training model...')
            with tqdm(range(config.n_epochs)) as pbar:
                for epoch in pbar:
                    pbar.set_description(f'Epoch {epoch}')
                    epoch_loss = 0
                    trunctuator = 0
                    states = model.model.get_starting_states(1, None) # train_dataset[0][1].unsqueeze(0)
                    seq_loss = 0
                    for x, y in train_dataloader:

                        outputs, states = model.model(x, states)

                        loss = model.criterion(outputs, y)
                        seq_loss = seq_loss + loss

                        epoch_loss += loss.item()

                        trunctuator += 1
                        if trunctuator == config.seq_len:
                            model.optimizer.zero_grad(set_to_none=True)
                            seq_loss = seq_loss / config.seq_len
                            seq_loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 4)
                            model.optimizer.step()
                            states = states.detach()
                            trunctuator = 0
                            seq_loss = 0

                        if torch.any(torch.isnan(loss)):
                            print('NAN Loss!')

                    val_loss, val_losses = evaluate_model(model, testsets, device, config)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wandb.run.summary['best_epoch'] = epoch
                        wandb.run.summary['best_val_loss'] = best_val_loss
                    wandb.run.summary['used_epochs'] = epoch

                    lr = model.scheduler.get_last_lr()[0]
                    model.scheduler.step(val_loss)  # Update the learning rate after each epoch
                    epoch_loss = epoch_loss / len(train_dataset)

                    pbar.set_postfix({'lr': lr, 'train_loss': epoch_loss, 'val_loss': val_loss})

                    # print('Total val loss:', val_loss)
                    log = {f'val_loss/{(config.recordings + config.test_recordings)[set_id]}': loss for set_id, loss in
                           enumerate(val_losses)}
                    log['total_val_loss'] = val_loss
                    log['train_loss'] = epoch_loss
                    log['lr'] = lr
                    wandb.log(log, step=epoch)

                    if early_stopper.early_stop(val_loss):
                        break

    else:
        cv2.namedWindow("Mac Camera")
        vc = cv2.VideoCapture(0)
        frame = vc.read()[0]
        width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)

        vc = WebcamVideoStream(src=0)
        anglesHelper = AnglesHelper()




        body_model_path = 'models/mediapipe/pose_landmarker_lite.task'
        hands_model_path = 'models/mediapipe/hand_landmarker.task'
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = mp.tasks.vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=hands_model_path)
                                                        , num_hands=2, running_mode=VisionRunningMode.VIDEO)
        hand_models = {'Right': mp.tasks.vision.HandLandmarker.create_from_options(options),
                       'Left': mp.tasks.vision.HandLandmarker.create_from_options(options)}

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

        frame_id = 0
        scales = (width, height, width)
        roi_half_size = 100
        sides = [args.intact_hand]

        joints_df = pd.DataFrame(index=range(calibration_frames), columns=columns)
        # sleep(1)
        check_time = time()
        vc.start()
        print('Calibrating...')
        for i in tqdm(range(calibration_frames)):
            print(1 / (time() - check_time))
            check_time = time()

            mp_image = vc.read()[0]

            frame_time = int(time() * 1000)
            body_results = body_model.detect_for_video(mp_image, frame_time).pose_landmarks  # todo check
            if len(body_results) == 0:
                continue

            for landmark_name in body_cols:
                joints_df.loc[i, ('Body', landmark_name, 'x')] = int(
                    body_results[0][pose.PoseLandmark[landmark_name]].x * scales[0])
                joints_df.loc[i, ('Body', landmark_name, 'y')] = int(
                    body_results[0][pose.PoseLandmark[landmark_name]].y * scales[1])
                joints_df.loc[i, ('Body', landmark_name, 'z')] = int(
                    body_results[0][pose.PoseLandmark[landmark_name]].z * scales[2])

            for side in sides:
                wrist = [joints_df.loc[i, ('Body', f'{side.upper()}_WRIST', 'x')],
                         joints_df.loc[i, ('Body', f'{side.upper()}_WRIST', 'y')]]

                x_start = 0
                x_end = scales[0]
                y_start = 0
                y_end = scales[1]

                hands_results = hand_models[side].detect_for_video(mp_image, frame_time).hand_landmarks  # todo check
                if len(hands_results) == 0:
                    continue
                elif len(hands_results) == 1:
                    hand_id = 0

                else:
                    x = joints_df.loc[i, ('Body', f'{side.upper()}_WRIST', 'x')]
                    y = joints_df.loc[i, ('Body', f'{side.upper()}_WRIST', 'y')]
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
                    joints_df.loc[i, (side, landmark_name, 'x')] = x_start + hands_results[hand_id][
                        hands.HandLandmark[landmark_name]].x * (x_end - x_start)
                    joints_df.loc[i, (side, landmark_name, 'y')] = y_start + hands_results[hand_id][
                        hands.HandLandmark[landmark_name]].y * (y_end - y_start)
                    joints_df.loc[i, (side, landmark_name, 'z')] = x_start + hands_results[hand_id][
                        hands.HandLandmark[landmark_name]].z * (x_end - x_start)
        vc.stop()
        joints_df = joints_df.fillna(0)
        joints_df = update_left_right(joints_df)
        average_upper_arm_length = {}
        average_forearm_length = {}
        for side in sides:
        # get upper arm length and forearm length:
            upper_arm_lengths = []
            forearm_lengths = []
            for i in range(calibration_frames):
                shoulder = joints_df.loc[i, (side, 'SHOULDER', ['x', 'y'])].values
                elbow = joints_df.loc[i, (side, 'ELBOW', ['x', 'y'])].values
                wrist = joints_df.loc[i, (side, 'WRIST', ['x', 'y'])].values

                upper_arm_lengths.append(np.linalg.norm(shoulder - elbow))
                forearm_lengths.append(np.linalg.norm(elbow - wrist))
            average_upper_arm_length[side] = np.mean(np.array(upper_arm_lengths))
            average_forearm_length[side] = np.mean(np.array(forearm_lengths))


        joints_df = pd.DataFrame(index=[0], columns=columns)

        wandb.init(mode=config.wandb_mode, project=config.wandb_project, name=config.name, config=config)
        config = wandb.config

        model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features), output_size=len(config.targets),
                                           **config)
        model.to(device)
        model.train()
        # if config.model_type == 'ActivationAndBiophys':  # todo
        #     for param in model.model.biophys_model.parameters():
        #         param.requires_grad = False if epoch < config.biophys_config['n_freeze_epochs'] else True

        best_val_loss = float('inf')
        print('Training model...')
        vc.start()
        with tqdm(range(config.n_epochs)) as pbar:
            for epoch in pbar:
                pbar.set_description(f'Epoch {epoch}')
                epoch_loss = 0
                trunctuator = 0
                states = model.model.get_starting_states(1, None) # train_dataset[0][1].unsqueeze(0)
                seq_loss = 0
                for i in range(epoch_len):
                    print(1 / (time() - check_time))
                    check_time = time()
                    mp_image, emg_timestep = vc.read()



                    frame_time = int(time() * 1000)
                    body_results = body_model.detect_for_video(mp_image, frame_time).pose_landmarks  # todo check
                    if len(body_results) == 0:
                        continue

                    for landmark_name in body_cols:
                        joints_df.loc[frame_id, ('Body', landmark_name, 'x')] = int(
                            body_results[0][pose.PoseLandmark[landmark_name]].x * scales[0])
                        joints_df.loc[frame_id, ('Body', landmark_name, 'y')] = int(
                            body_results[0][pose.PoseLandmark[landmark_name]].y * scales[1])
                        joints_df.loc[frame_id, ('Body', landmark_name, 'z')] = int(
                            body_results[0][pose.PoseLandmark[landmark_name]].z * scales[2])

                    for side in sides:
                        USE_ROI = False

                        wrist = [joints_df.loc[frame_id, ('Body', f'{side.upper()}_WRIST', 'x')],
                                 joints_df.loc[frame_id, ('Body', f'{side.upper()}_WRIST', 'y')]]
                        if USE_ROI:
                            x_start = max(0, wrist[0] - roi_half_size)
                            x_end = min(scales[0], wrist[0] + roi_half_size)
                            y_start = max(0, wrist[1] - roi_half_size)
                            y_end = min(scales[1], wrist[1] + roi_half_size)
                            cropped_frame = frame[y_start:y_end, x_start:x_end]
                            rgb_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)


                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_cropped_frame)
                        else:
                            x_start = 0
                            x_end = scales[0]
                            y_start = 0
                            y_end = scales[1]

                        hands_results = hand_models[side].detect_for_video(mp_image, frame_time).hand_landmarks  # todo check
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
                            joints_df.loc[frame_id, (side, landmark_name, 'x')] = x_start + hands_results[hand_id][
                                hands.HandLandmark[landmark_name]].x * (x_end - x_start)
                            joints_df.loc[frame_id, (side, landmark_name, 'y')] = y_start + hands_results[hand_id][
                                hands.HandLandmark[landmark_name]].y * (y_end - y_start)
                            joints_df.loc[frame_id, (side, landmark_name, 'z')] = x_start + hands_results[hand_id][
                                hands.HandLandmark[landmark_name]].z * (x_end - x_start)

                        joints_df = joints_df.fillna(0)
                    joints_df = update_left_right(joints_df)

                    for side in sides:

                        upper_arm = joints_df.loc[:, idx[side, 'ELBOW', slice(None)]].values - joints_df.loc[:,
                                                                                               idx[
                                                                                                   side, 'SHOULDER', slice(
                                                                                                       None)]].values
                        forearm = joints_df.loc[:, idx[side, 'WRIST', slice(None)]].values - joints_df.loc[:,
                                                                                             idx[side, 'ELBOW', slice(
                                                                                                 None)]].values
                        upper_arm = upper_arm.astype(np.float64)
                        forearm = forearm.astype(np.float64)

                        missing_len_upper_arm = average_upper_arm_length[side] ** 2 - upper_arm[:, 0] ** 2 - upper_arm[:,
                                                                                                       1] ** 2
                        missing_len_forearm = average_forearm_length[side] ** 2 - forearm[:, 0] ** 2 - forearm[:, 0] ** 2

                        missing_len_upper_arm = np.where(missing_len_upper_arm > 0, missing_len_upper_arm, 0)
                        missing_len_forearm = np.where(missing_len_forearm > 0, missing_len_forearm, 0)

                        upper_arm[:, 2] = np.sqrt(missing_len_upper_arm)
                        forearm[:, 2] = np.sqrt(missing_len_forearm)

                        joints_df.loc[:, idx['Body', f'{side.upper()}_ELBOW', 'z']] = joints_df.loc[:, idx[
                                                                                                           'Body', f'{side.upper()}_SHOULDER', 'z']].values + upper_arm[
                                                                                                                                                              :,
                                                                                                                                                              2] * -1
                        joints_df.loc[:, idx['Body', f'{side.upper()}_WRIST', 'z']] = joints_df.loc[:, idx[
                                                                                                           'Body', f'{side.upper()}_ELBOW', 'z']].values + forearm[
                                                                                                                                                           :,
                                                                                                                                                           2] * -1

                    joints_df = update_left_right(joints_df)
                    angles_df = anglesHelper.getArmAngles(joints_df)
                    # todo filtering
                    x = torch.tensor(emg_timestep[emg_channels], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                    outputs, states = model.model(x, states)

                    loss = model.criterion(outputs, y)
                    seq_loss = seq_loss + loss

                    epoch_loss += loss.item()

                    trunctuator += 1
                    if trunctuator == config.seq_len:
                        model.optimizer.zero_grad(set_to_none=True)
                        seq_loss = seq_loss / config.seq_len
                        seq_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 4)
                        model.optimizer.step()
                        states = states.detach()
                        trunctuator = 0
                        seq_loss = 0

                    if torch.any(torch.isnan(loss)):
                        print('NAN Loss!')

                val_loss, val_losses = evaluate_model(model, testsets, device, config)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wandb.run.summary['best_epoch'] = epoch
                    wandb.run.summary['best_val_loss'] = best_val_loss
                wandb.run.summary['used_epochs'] = epoch

                lr = model.scheduler.get_last_lr()[0]
                model.scheduler.step(val_loss)  # Update the learning rate after each epoch
                epoch_loss = epoch_loss / epoch_len
                pbar.set_postfix({'lr': lr, 'train_loss': epoch_loss, 'val_loss': val_loss})

                # print('Total val loss:', val_loss)
                log = {f'val_loss/{(config.recordings + config.test_recordings)[set_id]}': loss for set_id, loss in
                       enumerate(val_losses)}
                log['total_val_loss'] = val_loss
                log['train_loss'] = epoch_loss
                log['lr'] = lr
                wandb.log(log, step=epoch)



# import pybullet_utils.bullet_client as bc
# p = bc.BulletClient(connection_mode=pybullet.DIRECT)
import sys
import time
from time import sleep
from contextlib import contextmanager

multiplier = 1.05851325
offset = 0.72349796

from multiprocessing import Process, Event, Queue as MPQueue, Value
import pybullet as p

import mediapipe as mp
from mediapipe.python.solutions import pose, hands
import pandas as pd

pd.options.mode.copy_on_write = True
idx = pd.IndexSlice
from os.path import join

from helpers.utils import AnglesHelper
from helpers.EMGClass import EMG

import warnings
import signal

warnings.filterwarnings("ignore")
from time import time
import math
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import numpy as np
import torch
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

import cv2

from threading import Thread
from queue import Queue
import platform
from helpers.predict_utils import scale_data, rescale_data

FRAME_RATE = 60
frame_id = 0
system_name = platform.system()
PRINT = False


class InputThread(Thread):
    def __init__(self, src=0, queueSize=0, save=True, kE=None):
        super().__init__()
        self.stream = cv2.VideoCapture(src)
        self.outputQ = Queue(maxsize=queueSize)
        self.initialized = Event()
        self.emg = EMG()
        self.emg.startCommunication()
        self.fps = Value('f', 0)
        self.counter = 0
        if save:
            self.sampler = 1
        else:
            self.sampler = 3

        self.width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.input_fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.killEvent = Event()

    def run(self):
        self.initialized.set()
        while not self.killEvent.is_set():
            start_time = time()
            (self.grabbed, frame) = self.stream.read()
            emg_timestep = np.asarray(self.emg.normedEMG)
            if self.counter % self.sampler != 0:
                frame = None
            self.write_to_output((frame, emg_timestep))
            self.fps.value = 1 / (time() - start_time)
            self.counter += 1
        self.emg.shutdown()

    def write_to_output(self, data):
        if not self.outputQ.full():
            self.outputQ.put(data, block=False)
        else:
            print("InputThread output Queue is full", flush=True)
            pass

    def reset(self):
        while not self.outputQ.empty():  # Clear the queue
            self.outputQ.get()


class SaveThread(Thread):
    def __init__(self, inputQ, frame_size, save_path, kE=None):
        super().__init__()
        self.inputQ = inputQ
        self.save_path = save_path
        self.initialized = Event()
        self.frame_size = frame_size
        self.killEvent = Event()
        self.emg_data = []
        os.makedirs(save_path, exist_ok=True)

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(join(self.save_path, 'video.mp4'), fourcc, FRAME_RATE, self.frame_size)  # Adjust frame rate as needed
        self.initialized.set()

        while not self.killEvent.is_set():
            if not self.inputQ.empty():
                frame, emg_timestep = self.inputQ.get()
                if frame is not None:
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out.write(frame)
                    self.emg_data.append(emg_timestep)

        self.emg_data = np.array(self.emg_data)
        np.save(join(self.save_path, 'emg.npy'), self.emg_data)
        out.release()
        cv2.destroyAllWindows()



class JointsProcess(Process):
    def __init__(self, intact_hand, queueSize, outputQ, save_path, camera, save, calibration_frames, kE=None):
        super().__init__()
        self.outputQ = outputQ
        self.initialized = Event()
        self.intact_hand = intact_hand
        self.calibration_frames = calibration_frames
        self.fps = Value('f', 0)
        self.input_fps = Value('f', 0)
        self.killEvent = Event()
        self.queueSize = queueSize
        self.save_path = save_path
        self.camera = camera
        self.save = save

    def update_left_right(self, joints_df):
        for key, value in {'Left': 'LEFT', 'Right': 'RIGHT'}.items():
            for landmark_name in ['SHOULDER', 'ELBOW', 'HIP']:
                joints_df.loc[:, idx[key, landmark_name, slice(None)]] = joints_df.loc[:, idx
                                                                                          [
                                                                                              'Body', f'{value}_{landmark_name}', slice
                                                                                              (None)]].values

            joints_df.loc[:, idx[key, 'BODY_WRIST', ['x', 'y']]] = joints_df.loc[:,
                                                                   idx[key, 'WRIST', ['x', 'y']]].values
            joints_df.loc[:, idx[key, 'BODY_WRIST', 'z']] = joints_df.loc[:, idx['Body', f'{value}_WRIST', 'z']].values
        return joints_df

    def run(self):
        temp_vc = cv2.VideoCapture(self.camera)
        width = temp_vc.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = temp_vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
        temp_vc.release()

        vc = InputThread(src=self.camera, queueSize=self.queueSize, save=self.save)
        vc.start()
        vc.initialized.wait()

        body_model_path = 'models/mediapipe/pose_landmarker_lite.task'
        hands_model_path = 'models/mediapipe/hand_landmarker.task'
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = mp.tasks.vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=hands_model_path)
                                                        , num_hands=1, running_mode=VisionRunningMode.VIDEO) # todo num_hands
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

        scales = (width, height, width)
        side = self.intact_hand

        print('Calibrating...')
        joints_df = pd.DataFrame(columns=columns)
        indexer = 0
        with tqdm(total=self.calibration_frames) as pbar:
            while indexer < self.calibration_frames:
                frame = vc.outputQ.get()[0]
                if frame is None:
                    continue
                if system_name == 'Darwin':
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                frame_time = int(time() * 1000)
                body_results = body_model.detect_for_video(mp_image, frame_time).pose_landmarks  # todo check
                if len(body_results) == 0:
                    continue

                for landmark_name in body_cols:
                    joints_df.loc[indexer, ('Body', landmark_name, 'x')] = int(
                        body_results[0][pose.PoseLandmark[landmark_name]].x * scales[0])
                    joints_df.loc[indexer, ('Body', landmark_name, 'y')] = int(
                        body_results[0][pose.PoseLandmark[landmark_name]].y * scales[1])
                    joints_df.loc[indexer, ('Body', landmark_name, 'z')] = int(
                        body_results[0][pose.PoseLandmark[landmark_name]].z * scales[2])

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
                    x = joints_df.loc[indexer, ('Body', f'{side.upper()}_WRIST', 'x')]
                    y = joints_df.loc[indexer, ('Body', f'{side.upper()}_WRIST', 'y')]
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
                    joints_df.loc[indexer, (side, landmark_name, 'x')] = x_start + hands_results[hand_id][
                        hands.HandLandmark[landmark_name]].x * (x_end - x_start)
                    joints_df.loc[indexer, (side, landmark_name, 'y')] = y_start + hands_results[hand_id][
                        hands.HandLandmark[landmark_name]].y * (y_end - y_start)
                    joints_df.loc[indexer, (side, landmark_name, 'z')] = x_start + hands_results[hand_id][
                        hands.HandLandmark[landmark_name]].z * (x_end - x_start)
                indexer += 1
                pbar.update(1)
        # joints_df = joints_df.fillna(0)
        joints_df = self.update_left_right(joints_df)
        average_upper_arm_length = {}
        average_forearm_length = {}
        # get upper arm length and forearm length:
        upper_arm_lengths = []
        forearm_lengths = []
        for i in range(self.calibration_frames):
            shoulder = joints_df.loc[i, (side, 'SHOULDER', ['x', 'y'])].values
            elbow = joints_df.loc[i, (side, 'ELBOW', ['x', 'y'])].values
            wrist = joints_df.loc[i, (side, 'WRIST', ['x', 'y'])].values

            upper_arm_lengths.append(np.linalg.norm(shoulder - elbow))
            forearm_lengths.append(np.linalg.norm(elbow - wrist))
        average_upper_arm_length[side] = np.mean(np.array(upper_arm_lengths))
        average_forearm_length[side] = np.mean(np.array(forearm_lengths))

        joints_df = pd.DataFrame(index=[frame_id], columns=columns, dtype=np.float64)

        self.initialized.set()
        t1 = time()
        t2 = time()
        t3 = time()
        t4 = time()
        while not self.killEvent.is_set():
            frame, emg_timestep = vc.outputQ.get()
            if frame is None:
                self.write((None, emg_timestep))
                continue
            if system_name == 'Darwin':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            start_time = time()


            frame_time = int(time() * 1000)
            t1 = time()
            body_results = body_model.detect_for_video(mp_image, frame_time).pose_landmarks
            t2 = time()
            if PRINT:
                print('body: ', t2 - t1)

            if len(body_results) == 0:
                self.write((None, emg_timestep))
                continue

            for landmark_name in body_cols:
                joints_df.loc[frame_id, ('Body', landmark_name, 'x')] = int(
                    body_results[0][pose.PoseLandmark[landmark_name]].x * scales[0])
                joints_df.loc[frame_id, ('Body', landmark_name, 'y')] = int(
                    body_results[0][pose.PoseLandmark[landmark_name]].y * scales[1])
                joints_df.loc[frame_id, ('Body', landmark_name, 'z')] = int(
                    body_results[0][pose.PoseLandmark[landmark_name]].z * scales[2])

            if side == 'Right':
                x_start = 0
                x_end = int(scales[0] // 2)
            else:
                x_start = int(scales[0] // 2)
                x_end = int(scales[0])

            y_start = 0
            y_end = int(scales[1])

            frame = frame[:, x_start:x_end]

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if system_name != 'Darwin':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            hands_results = hand_models[side].detect_for_video(mp_image, frame_time).hand_landmarks
            t3 = time()
            if PRINT:
                print('hands: ', t3 - t2)
            if len(hands_results) == 0:
                self.write((None, emg_timestep))
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

            t4 = time()
            if PRINT:
                print('hand_id: ', t4 - t3)

            for landmark_name in hands.HandLandmark._member_names_:
                joints_df.loc[frame_id, (side, landmark_name, 'x')] = x_start + hands_results[hand_id][
                    hands.HandLandmark[landmark_name]].x * (x_end - x_start)
                joints_df.loc[frame_id, (side, landmark_name, 'y')] = y_start + hands_results[hand_id][
                    hands.HandLandmark[landmark_name]].y * (y_end - y_start)
                joints_df.loc[frame_id, (side, landmark_name, 'z')] = x_start + hands_results[hand_id][
                    hands.HandLandmark[landmark_name]].z * (x_end - x_start)

            # joints_df = joints_df.fillna(0)
            joints_df = self.update_left_right(joints_df)

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
            t5 = time()
            if PRINT:
                print('correction: ', t5 - t4)
            joints_df = self.update_left_right(joints_df)
            self.write((joints_df, emg_timestep))
            self.fps.value = 1 / (time() - start_time)
            self.input_fps.value = vc.fps.value
        vc.killEvent.set()
        vc.join()

    def write(self, data):
        if not self.outputQ.full():
            self.outputQ.put(data, block=False)
        else:
            print("JointProcess output Queue full", flush=True)
            pass
    def reset(self):
        while not self.outputQ.empty():
            self.outputQ.get()


class AnglesProcess(Process):
    def __init__(self, intact_hand, queueSize=0, inputQ=None, kE=None):
        super().__init__()
        self.inputQ = inputQ
        self.outputQ = MPQueue(queueSize)
        self.initialized = Event()
        self.intact_hand = intact_hand
        self.fps = Value('f', 0)
        self.killEvent = Event()


    def run(self):
        max_interpolation_steps = 30
        anglesHelper = AnglesHelper()
        sides = [self.intact_hand]
        emg_buffer = []

        joints_df = None
        while joints_df is None: # make sure the first frame is not None
            joints_df, emg_timestep = self.inputQ.get()
        angles_df = anglesHelper.getArmAngles(joints_df, sides)
        angles_df = angles_df.reindex(range(max_interpolation_steps))
        self.initialized.set()

        while not self.killEvent.is_set():
            joints_df, emg_timestep = self.inputQ.get()
            emg_buffer.append(emg_timestep)
            if joints_df is None:
                continue
            start_time = time()
            steps = len(emg_buffer)
            if steps > max_interpolation_steps:
                exit(f'More than {max_interpolation_steps} interpolation steps ({steps})')
            angles_df.loc[1:, :] = np.nan
            angles_df.loc[steps, :] = anglesHelper.getArmAngles(joints_df, sides).loc[frame_id, :]
            interpolated = angles_df.interpolate(method='index', limit_area='inside')
            angles_df.loc[0, :] = angles_df.loc[steps, :]
            interpolated = scale_data(interpolated, self.intact_hand)
            for i in range(0, steps):
                self.write((interpolated.loc[i+1, :], emg_buffer[i]))
            emg_buffer = []
            self.fps.value = 3 / (time() - start_time)

    def write(self, data):
        if not self.outputQ.full():
            self.outputQ.put(data, block=False)
        else:
            print("AnglesProcess output queue full", flush=True)
            pass

    def reset(self):
        while not self.outputQ.empty():
            self.outputQ.get()


class VisualizeProcess(Process):
    def __init__(self, intact_hand, inputQ=None, kE=None):
        super().__init__()
        self.intact_hand = intact_hand
        self.inputQ = inputQ
        self.initialized = Event()
        self.fps = Value('f', 0)
        self.killEvent = Event()


    def run(self):
        intact_hand = self.intact_hand

        joint_ids = {
            'elbow': 0,
            'wrist_rotation': 1,
            'wrist_flexion': 2,
            'index': (4, 5, 6),
            'middle': (7, 8, 9),
            'ring': (10, 11, 12),
            'pinky': (13, 14, 15),
            'thumb': (16, 17, 18)
        }

        @contextmanager
        def suppress_stdout():
            fd = sys.stdout.fileno()

            def _redirect_stdout(to):
                sys.stdout.close()  # + implicit flush()
                os.dup2(to.fileno(), fd)  # fd writes to 'to' file
                sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

            with os.fdopen(os.dup(fd), "w") as old_stdout:
                with open(os.devnull, "w") as file:
                    _redirect_stdout(to=file)
                try:
                    yield  # allow code to be run with the redirected stdout
                finally:
                    _redirect_stdout(to=old_stdout)  # restore stdout.
                    # buffering and flags such as
                    # CLOEXEC may be different

        def init_physics_client():
            physicsClient = p.connect(p.GUI)
            p.setGravity(0, 0, -9.81)

            handStartPos = [0, 0, 0]
            handStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
            urdf_path = "URDF/ability_hand_left_large-wrist.urdf" if intact_hand == 'Left' else "URDF/ability_hand_right_large-wrist.urdf"

            # with suppress_stdout():
            target_hand = p.loadURDF(urdf_path, handStartPos, handStartOrientation,
                                     flags=p.URDF_USE_SELF_COLLISION, useFixedBase=True)
            pred_hand = p.loadURDF(urdf_path, handStartPos, handStartOrientation,
                                   flags=p.URDF_USE_SELF_COLLISION, useFixedBase=True)

            for i in range(1, p.getNumJoints(target_hand)):
                p.changeVisualShape(target_hand, i, rgbaColor=[0, 1, 0, 0.7])

            for i in range(1, p.getNumJoints(pred_hand)):
                p.changeVisualShape(pred_hand, i, rgbaColor=[1, 0, 0, 0.7])

            camera_distance = 1.34
            camera_yaw = 223
            camera_pitch = -25
            camera_target_position = [0.59, -0.65, -0.3]
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)
            p.stepSimulation()

            return target_hand, pred_hand


        def move_finger(hand, finger, angle):
            id0, id1 = joint_ids[finger][0], joint_ids[finger][1]
            p.setJointMotorControl2(hand, id0, p.POSITION_CONTROL, targetPosition=angle)
            p.setJointMotorControl2(hand, id1, p.POSITION_CONTROL, targetPosition=angle * multiplier + offset)

        # Initialize physics client within the process
        target_hand, pred_hand = init_physics_client()
        self.initialized.set()  # Signal that the initialization is complete

        while not self.killEvent.is_set():
            target_angles, pred_angles = self.inputQ.get()
            time_start = time()

            target_angles = rescale_data(target_angles, self.intact_hand)
            pred_angles = rescale_data(pred_angles, self.intact_hand)


            move_finger(target_hand, 'index', target_angles.loc[(intact_hand, 'indexAng')])
            move_finger(target_hand, 'middle', target_angles.loc[(intact_hand, 'midAng')])
            move_finger(target_hand, 'ring', target_angles.loc[(intact_hand, 'ringAng')])
            move_finger(target_hand, 'pinky', target_angles.loc[(intact_hand, 'pinkyAng')])

            angle = target_angles.loc[(intact_hand, 'thumbInPlaneAng')]
            p.setJointMotorControl2(target_hand, joint_ids['thumb'][0], p.POSITION_CONTROL, targetPosition=angle)
            angle = target_angles.loc[(intact_hand, 'thumbOutPlaneAng')]
            p.setJointMotorControl2(target_hand, joint_ids['thumb'][1], p.POSITION_CONTROL, targetPosition=angle)

            angle = target_angles.loc[(intact_hand, 'wristRot')]
            p.setJointMotorControl2(target_hand, joint_ids['wrist_rotation'], p.POSITION_CONTROL, targetPosition=angle)
            angle = target_angles.loc[(intact_hand, 'wristFlex')]
            p.setJointMotorControl2(target_hand, joint_ids['wrist_flexion'], p.POSITION_CONTROL, targetPosition=angle)

            move_finger(pred_hand, 'index', pred_angles.loc[(intact_hand, 'indexAng')])
            move_finger(pred_hand, 'middle', pred_angles.loc[(intact_hand, 'midAng')])
            move_finger(pred_hand, 'ring', pred_angles.loc[(intact_hand, 'ringAng')])
            move_finger(pred_hand, 'pinky', pred_angles.loc[(intact_hand, 'pinkyAng')])

            angle = pred_angles.loc[(intact_hand, 'thumbInPlaneAng')]
            p.setJointMotorControl2(pred_hand, joint_ids['thumb'][0], p.POSITION_CONTROL, targetPosition=angle)
            angle = pred_angles.loc[(intact_hand, 'thumbOutPlaneAng')]
            p.setJointMotorControl2(pred_hand, joint_ids['thumb'][1], p.POSITION_CONTROL, targetPosition=angle)

            angle = pred_angles.loc[(intact_hand, 'wristRot')]
            p.setJointMotorControl2(pred_hand, joint_ids['wrist_rotation'], p.POSITION_CONTROL, targetPosition=angle)
            angle = pred_angles.loc[(intact_hand, 'wristFlex')]
            p.setJointMotorControl2(pred_hand, joint_ids['wrist_flexion'], p.POSITION_CONTROL, targetPosition=angle)
            p.stepSimulation()

            self.fps.value = (1 / (time() - time_start))
        print('VisualizeProcess killed', flush=True)

class ProcessManager:
    def __init__(self):

        self.processes = []
        self.killEvent = Event()

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        for process in self.processes:
            process.killEvent.set()
            sleep(1)

        self.killEvent.set()
        sleep(5)
        # sys.exit(0)

    def manage_process(self, process):
        process.start()
        process.initialized.wait()
        self.processes.append(process)


if __name__ == '__main__':
    mediapipe_test = True
    if mediapipe_test:
        print(system_name)
        cap = cv2.VideoCapture(2)  # 0 for the default camera, or provide a video file path


        body_model_path = 'models/mediapipe/pose_landmarker_lite.task'
        hands_model_path = 'models/mediapipe/hand_landmarker.task'
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = mp.tasks.vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=hands_model_path)
                                                        , num_hands=2,
                                                        running_mode=VisionRunningMode.VIDEO)
        hand_models = {'Right': mp.tasks.vision.HandLandmarker.create_from_options(options),
                       'Left': mp.tasks.vision.HandLandmarker.create_from_options(options)}

        detector = hand_models['Right']
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=body_model_path),
            running_mode=VisionRunningMode.VIDEO)
        body_model = PoseLandmarker.create_from_options(options)

        # Create a VideoCapture object
        before_time = time()

        from mediapipe import solutions
        from mediapipe.framework.formats import landmark_pb2
        import numpy as np

        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


        def draw_landmarks_on_image(rgb_image, detection_result):
            hand_landmarks_list = detection_result.hand_landmarks
            handedness_list = detection_result.handedness
            annotated_image = np.copy(rgb_image)

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                handedness = handedness_list[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style())

                # Get the top left corner of the detected hand's bounding box.
                height, width, _ = annotated_image.shape
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height) - MARGIN

                # Draw handedness (left or right hand) on the image.
                cv2.putText(annotated_image, f"{handedness[0].category_name}",
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

            return annotated_image


        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            frame = frame[:, :frame.shape[1]//2, :]  # right hand
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if system_name != 'Darwin':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            detection_result = detector.detect_for_video(image, int(time() * 1000))
            annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
            # if system_name == 'Darwin':
            #     annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR) # todo
            cv2.imshow('show', annotated_image)

            if not ret:
                break


            # cv2.imshow('Frame', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture object and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()


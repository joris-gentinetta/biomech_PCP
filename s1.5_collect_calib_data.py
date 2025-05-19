#!/usr/bin/env python3
import argparse
import os
import time
import threading
import numpy as np
import cv2
from os.path import join
from helpers.hand_poses import hand_poses
from psyonicHand import psyonicArm
from scipy.interpolate import CubicSpline
from helpers.EMGClass import EMG

def start_raw_emg_recorder(base_dir, enable_video=False):
    """
    Start background threads to record raw EMG and (optionally) webcam video.
    Returns: stop_event, emg_thread, video_thread (or None), raw_history, raw_timestamps
    """
    # EMG setup
    emg = EMG()  # connect directly to ADS1299 via EMGClass
    emg.startCommunication()

    raw_history = []
    raw_timestamps = []
    stop_event = threading.Event()

    # Optional video recorder thread
    video_thread = None
    if enable_video:
        def video_loop():
            cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = join(base_dir, 'webcam.mp4')
            # query camera resolution
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                time.sleep(1.0 / fps)
            cap.release()
            writer.release()

        video_thread = threading.Thread(target=video_loop, daemon=True)
        video_thread.start()

    # EMG capture thread
    def capture_loop():
        # wait for first EMG packet
        while getattr(emg, 'OS_time', None) is None:
            time.sleep(1e-3)
        last_time = emg.OS_time
        while not stop_event.is_set():
            t = emg.OS_time
            if t is not None and t > last_time:
                raw_history.append(list(emg.rawEMG))
                raw_timestamps.append(t)
                last_time = t
            time.sleep(1e-4)
        emg.exitEvent.set()

    emg_thread = threading.Thread(target=capture_loop, daemon=True)
    emg_thread.start()

    return stop_event, emg_thread, video_thread, raw_history, raw_timestamps

def main():
    parser = argparse.ArgumentParser(
        description="Record synchronized EMG + prosthetic hand pose data"
    )
    parser.add_argument("--person_id", "-p", required=True,
                        help="Person ID (folder under data/)")
    parser.add_argument("--movement", "-m", required=True,
                        help="Movement name (subfolder under recordings/)")
    parser.add_argument("--out_root", "-o", default="data",
                        help="Root data directory (default: ./data)")
    parser.add_argument("--no_emg", action="store_true",
                        help="Disable EMG capture for testing without hardware")
    parser.add_argument("--no_prosthesis", action="store_true",
                        help="Disable prosthetic arm control; EMG-only recording")
    parser.add_argument("--hand_side", "-s", choices=["left", "right"],
                        default="left", help="Side of the prosthetic hand")
    parser.add_argument("--sync_iterations", type=int, default=1,
                        help="Warm-up sync iterations (default: 1)")
    parser.add_argument("--record_iterations", "-r", type=int, default=1,
                        help="Number of recording iterations (default: 1)")
    parser.add_argument("--video", action="store_true",
                        help="Enable simultaneous webcam video recording")
    args = parser.parse_args()

    # Prepare output directory
    base_dir = os.path.join(
        args.out_root,
        args.person_id,
        "recordings",
        args.movement,
        "experiments",
        "1"
    )
    os.makedirs(base_dir, exist_ok=True)

    # Start raw EMG (and optional video) recorder in background
    if not args.no_emg:
        stop_event, raw_thread, video_thread, raw_history, raw_timestamps = \
            start_raw_emg_recorder(base_dir, enable_video=args.video)
    else:
        stop_event = raw_thread = video_thread = None
        raw_history = []
        raw_timestamps = []

    # Raw-EMG-only mode
    if args.no_prosthesis:
        raw_seconds = 20.0
        print(f"Raw-only mode: recording raw EMG for {raw_seconds:.1f} seconds…")
        try:
            time.sleep(raw_seconds)
        except KeyboardInterrupt:
            print("Interrupted early by user")
        stop_event.set()
        raw_thread.join()
        if video_thread:
            video_thread.join()
        if raw_history:
            np.save(join(base_dir, "raw_emg.npy"), np.vstack(raw_history))
            np.save(join(base_dir, "raw_timestamps.npy"),
                    np.array(raw_timestamps))
            print(f"Saved raw_emg.npy with {len(raw_history)} samples.")
        else:
            print("No raw data captured.")
        print("Done.")
        return

    # Filtered EMG subscriber for control loop
    if not args.no_emg:
        emg_ctrl = EMG(
            socketAddr='tcp://127.0.0.1:1236',
            usedChannels=[0, 1, 2, 4, 5, 8, 10, 11]
        )
        emg_ctrl.startCommunication()
        print("Filtered EMG subscriber connected for control.")
    else:
        emg_ctrl = None
        print("EMG capture disabled; running in no-EMG mode.")

    # Initialize prosthetic arm
    arm = psyonicArm(hand=args.hand_side)
    arm.initSensors()
    arm.startComms()

    # Validate movement
    if args.movement not in hand_poses:
        print(f"Unknown movement '{args.movement}'. Available: {list(hand_poses.keys())}")
        arm.close()
        return
    pose = hand_poses[args.movement]

    # Build and smooth trajectory
    neutral = np.array([2, 2, 2, 2, 2, -2])
    if isinstance(pose[0], (list, np.ndarray)):
        key_poses = [neutral] + [np.array(p) for p in pose] + [neutral]
    else:
        key_poses = [neutral, np.array(pose), neutral]
    total_dur = 2.0
    steps = 120
    times = np.linspace(0, total_dur, len(key_poses))
    interp = np.linspace(0, total_dur, steps)
    traj = CubicSpline(times, np.vstack(key_poses), axis=0)(interp)

    # Enforce constant speed
    diffs = np.diff(traj, axis=0)
    dist = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0], np.cumsum(dist)])
    s_u = np.linspace(0, s[-1], len(s))
    const_traj = np.zeros_like(traj)
    idx = np.clip(np.searchsorted(s, s_u) - 1, 0, len(s) - 2)
    for i, su in enumerate(s_u):
        i0 = idx[i]
        ds = s[i0 + 1] - s[i0]
        α = (su - s[i0]) / ds if ds > 0 else 0
        const_traj[i] = traj[i0] * (1 - α) + traj[i0 + 1] * α
    smooth_traj = const_traj

    # Warm-up sync
    for _ in range(args.sync_iterations):
        arm.mainControlLoop(posDes=smooth_traj, period=10, emg=emg_ctrl)

    # Recording loop
    all_records = []
    headers = None
    joint_names = ['index', 'middle', 'ring', 'pinky', 'thumbFlex', 'thumbRot']
    angle_cols = [f"{j}_Pos" for j in joint_names]

    for itr in range(1, args.record_iterations + 1):
        print(f"Starting iteration {itr}/{args.record_iterations}...")
        arm.resetRecording()
        arm.recording = True
        arm.mainControlLoop(posDes=smooth_traj, period=10, emg=emg_ctrl)
        arm.recording = False

        raw_data = arm.recordedData
        if headers is None:
            headers = raw_data[0]
            angle_idxs = [headers.index(c) for c in angle_cols]
        data_rows = raw_data[1:]
        all_records.extend(data_rows)

    # Stop raw EMG (and video) recorder
    if not args.no_emg:
        stop_event.set()
        raw_thread.join()
        if video_thread:
            video_thread.join()

    # Save EMG + angles
    rec = np.array(all_records, dtype=float)
    ts = rec[:, 0]
    emg_data = rec[:, 1:1 + len(emg_ctrl.usedChannels)] if emg_ctrl else np.empty((0, 0))
    angles = rec[:, angle_idxs]
    np.save(join(base_dir, "emg.npy"), emg_data)
    np.save(join(base_dir, "emg_timestamps.npy"), ts)
    np.save(join(base_dir, "angles.npy"), angles)
    np.save(join(base_dir, "angle_timestamps.npy"), ts)

    # Save raw EMG
    if raw_history:
        np.save(join(base_dir, "raw_emg.npy"), np.vstack(raw_history))
        np.save(join(base_dir, "raw_timestamps.npy"), np.array(raw_timestamps))
        print(f"Saved raw_emg.npy with {len(raw_history)} samples.")

    arm.close()
    if emg_ctrl:
        emg_ctrl.exitEvent.set()

if __name__ == "__main__":
    main()

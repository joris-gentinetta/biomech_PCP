#!/usr/bin/env python3
import argparse
import os
import time
import threading
import numpy as np
import cv2
import yaml
from os.path import join
from helpers.hand_poses import hand_poses
from psyonicHand import psyonicArm
from scipy.interpolate import CubicSpline
from helpers.EMGClass import EMG
from scipy.signal import butter, filtfilt
from helpers.BesselFilter import BesselFilterArr
def filter_emg_pipeline_bessel(raw_emg, fs, noise_level=None):
    # raw_emg: shape [channels, samples] (transpose if needed)
    num_channels = raw_emg.shape[0]

    # 1. Bandstop (powerline)
    notch = BesselFilterArr(numChannels=num_channels, order=8, critFreqs=[58,62], fs=fs, filtType='bandstop')
    emg = notch.filter(raw_emg)
    # 2. Highpass (20 Hz)
    hp = BesselFilterArr(numChannels=num_channels, order=4, critFreqs=20, fs=fs, filtType='highpass')
    emg = hp.filter(emg)
    # 3. Rectification
    emg = np.abs(emg)
    # 4. Noise subtraction and clipping
    if noise_level is not None:
        emg = np.clip(emg - noise_level[:, None], 0, None)
    # 5. Lowpass (envelope, 3 Hz)
    lp = BesselFilterArr(numChannels=num_channels, order=4, critFreqs=3, fs=fs, filtType='lowpass')
    emg = lp.filter(emg)

    return emg  # same shape as input


def robust_mvc(mvc_data, min_count=3, nbins=500):
    # mvc_data: [samples, channels]
    robust_max = []
    for ch in range(mvc_data.shape[1]):
        channel_data = mvc_data[:, ch]
        # Use a histogram to group "similar" values
        hist, bin_edges = np.histogram(channel_data, bins=nbins)
        # Find all bins with count >= min_count
        valid_bins = np.where(hist >= min_count)[0]
        if len(valid_bins) == 0:
            # Fallback: just use the overall max
            robust_max.append(np.max(channel_data))
        else:
            # Use the highest value bin
            max_bin = valid_bins[-1]
            robust_val = bin_edges[max_bin + 1]  # right edge of the bin
            robust_max.append(robust_val)
    return np.array(robust_max)

def calibrate_emg(base_dir, rest_time=5, mvc_time=10):
    print("EMG Calibration Routine")
    emg = EMG()
    emg.startCommunication()

    print("1. Relax for baseline noise recording")
    time.sleep(2)
    emg_rawHistory = []
    rest_timestamps = []
    emg.exitEvent.clear()

    # Record REST data
    t0 = time.time()
    while time.time() - t0 < rest_time:
        if hasattr(emg, 'rawEMG'):
            emg_rawHistory.append(list(emg.rawEMG))
            rest_timestamps.append(time.time())
        time.sleep(0.001)
    rest_data = np.vstack(emg_rawHistory)
    rest_timestamps = np.array(rest_timestamps)

    # --- FILTER REST DATA ---
    if len(rest_timestamps) > 1:
        sf_rest = (len(rest_timestamps) - 1) / (rest_timestamps[-1] - rest_timestamps[0])
    else:
        sf_rest = 1000  # fallback

    # Filtering: do NOT subtract noise yet!
    filtered_rest = filter_emg_pipeline_bessel(rest_data.T, sf_rest)
    noise_levels = np.mean(filtered_rest, axis=1)

    # -- MVC COLLECTION --
    input("2. Prepare for MVC (5x full fist and full hand open in 10s), press enter when ready")
    time.sleep(1)
    emg_rawHistory = []
    mvc_timestamps = []
    emg.exitEvent.clear()

    # Record MVC data
    t0 = time.time()
    while time.time() - t0 < mvc_time:
        if hasattr(emg, 'rawEMG'):
            emg_rawHistory.append(list(emg.rawEMG))
            mvc_timestamps.append(time.time())
        time.sleep(0.001)
    mvc_data = np.vstack(emg_rawHistory)
    mvc_timestamps = np.array(mvc_timestamps)

    if len(mvc_timestamps) > 1:
        sf_mvc = (len(mvc_timestamps) - 1) / (mvc_timestamps[-1] - mvc_timestamps[0])
    else:
        sf_mvc = 1000

    # --- FILTER MVC DATA and subtract noise floor ---
    filtered_mvc = filter_emg_pipeline_bessel(mvc_data.T, sf_mvc)
    filtered_mvc = np.clip(filtered_mvc - noise_levels[:, None], 0, None)

    # --- Compute robust max (per channel) ---
    max_vals = robust_mvc(filtered_mvc.T, min_count=3, nbins=500)

    emg.exitEvent.set()
    time.sleep(0.5)

    # --- Store result in scaling.yaml ---
    scaling_dict = {
        'maxVals': max_vals.tolist(),
        'noiseLevels': noise_levels.tolist()
    }
    yaml_path = os.path.join(base_dir, 'scaling.yaml')
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(scaling_dict, f)

    print(f"Saved scaling.yaml to {yaml_path}")
    print(f"Max vals and noise lvl: {max_vals},    {noise_levels}")

def start_raw_emg_recorder(base_dir, enable_video=False, sync_event=None):
    """
    Start background threads to record raw EMG and (optionally) webcam video.
    Returns: stop_event, emg_thread, video_thread (or None), raw_history, raw_timestamps, video_timestamps
    """
    emg = EMG()  # connect directly to ADS1299 via EMGClass (records all 16 channels by default)
    emg.startCommunication()

    raw_history = []
    raw_timestamps = []
    video_timestamps = []
    stop_event = threading.Event()

    # Optional video recorder thread with sync_event
    video_thread = None
    def video_loop():
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = join(base_dir, 'webcam.mp4')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        # Wait for main event to start
        if sync_event is not None:
            sync_event.wait()
        video_first_time = None
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            if video_first_time is None:
                video_first_time = now
            video_timestamps.append(now - video_first_time)
            writer.write(frame)
            time.sleep(1.0 / fps)
        cap.release()
        writer.release()

    if enable_video:
        video_thread = threading.Thread(target=video_loop, daemon=True)
        video_thread.start()

    # EMG capture thread with sync_event
    def capture_loop():
        # wait for first EMG packet
        while getattr(emg, 'OS_time', None) is None:
            time.sleep(1e-3)
        # Wait for main event to start
        if sync_event is not None:
            sync_event.wait()
        first_emg_time = emg.OS_time
        last_time = emg.OS_time
        while not stop_event.is_set():
            t = emg.OS_time
            if t is not None and t > last_time:
                raw_history.append(list(emg.rawEMG))
                raw_timestamps.append(t - first_emg_time)
                last_time = t
            time.sleep(1e-4)
        emg.exitEvent.set()

    emg_thread = threading.Thread(target=capture_loop, daemon=True)
    emg_thread.start()

    return stop_event, emg_thread, video_thread, raw_history, raw_timestamps, video_timestamps

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
    parser.add_argument("--sync_iterations", type=int, default=5,
                        help="Warm-up sync iterations (default: 1)")
    parser.add_argument("--record_iterations", "-r", type=int, default=20,
                        help="Number of recording iterations (default: 1)")
    parser.add_argument("--video", action="store_true",
                        help="Enable simultaneous webcam video recording")
    parser.add_argument("--calibrate_emg", action="store_true",
                    help="Run EMG calibration (noise level and MVC detection)")
    args = parser.parse_args()

    # Prepare output directory
    exp_parent = os.path.join(
        args.out_root,
        args.person_id,
        "recordings",
        args.movement,
        "experiments"
    )

    exp_idx = 1
    while os.path.exists(os.path.join(exp_parent, str(exp_idx))):
        exp_idx += 1
    base_dir = os.path.join(exp_parent, str(exp_idx))
    os.makedirs(base_dir, exist_ok=True)

    # Synchronization event for EMG and video threads
    sync_event = threading.Event() if (not args.no_emg or args.video) else None

    # Start raw EMG (and optional video) recorder in background, pass sync_event
    if not args.no_emg:
        stop_event, raw_thread, video_thread, raw_history, raw_timestamps, video_timestamps = \
            start_raw_emg_recorder(base_dir, enable_video=args.video, sync_event=sync_event)
    else:
        stop_event = raw_thread = video_thread = None
        raw_history = []
        raw_timestamps = []
        video_timestamps = []

    if args.calibrate_emg:
        calibrate_emg(base_dir, rest_time=5, mvc_time=10)
        return
    

    # Raw-EMG-only mode (no prosthesis movement, just record)
    if args.no_prosthesis:
        raw_seconds = 20.0
        print(f"Raw-only mode: recording raw EMG for {raw_seconds:.1f} seconds…")
        if sync_event is not None:
            sync_event.set()  # Start EMG/video capture immediately in raw-only mode
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
            if video_timestamps:
                np.save(join(base_dir, "video_timestamps.npy"), np.array(video_timestamps))
            print(f"Saved raw_emg.npy with {len(raw_history)} samples.")
        else:
            print("No raw data captured.")
        print("Done.")
        return

    # ----- PROSTHESIS MODE -----
    # Initialize prosthetic arm, but DO NOT use EMG for control
    arm = psyonicArm(hand=args.hand_side)
    arm.initSensors()
    arm.startComms()

    # Validate movement
    if args.movement not in hand_poses:
        print(f"Unknown movement '{args.movement}'. Available: {list(hand_poses.keys())}")
        arm.close()
        return
    pose = hand_poses[args.movement]

    if args.movement == "indexFlDigitsEx" and isinstance(pose, list) and len(pose) == 2:
        pos1 = np.array(pose[0])
        pos2 = np.array(pose[1])
        total_dur = 2.0  # total duration (seconds)
        steps = 500      # number of trajectory steps

        key_poses = [pos1, pos2, pos1]
        times = [0, total_dur / 2, total_dur]

        # Create symmetric, smooth out-and-back trajectory
        smooth_traj = CubicSpline(
            times, 
            np.vstack(key_poses), 
            axis=0, 
            bc_type='clamped'  # ensures smooth (zero velocity) at endpoints
        )(np.linspace(0, total_dur, steps))

    else:
        # Build and smooth trajectory
        neutral = np.array([2, 2, 2, 2, 2, -2])
        if isinstance(pose[0], (list, np.ndarray)):
            key_poses = [neutral] + [np.array(p) for p in pose] + [neutral]
        else:
            key_poses = [neutral, np.array(pose), neutral]
        total_dur = 2.0
        steps = 600
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
            alpha = (su - s[i0]) / ds if ds > 0 else 0
            const_traj[i] = traj[i0] * (1 - alpha) + traj[i0 + 1] * alpha
        smooth_traj = const_traj

    # --- CLIP THE TRAJECTORY TO SAFE RANGES ---
    joint_mins = np.array([0, 0, 0, 0, 0, -120])
    joint_maxs = np.array([120, 120, 120, 120, 120, 0])
    smooth_traj = np.clip(smooth_traj, joint_mins, joint_maxs)

    # Warm-up sync
    for _ in range(args.sync_iterations):
        arm.mainControlLoop(posDes=smooth_traj, period=10, emg=None)

    # --- Start synchronized recording! ---
    if sync_event is not None:
        sync_event.set()  # Signal all threads to start actual recording

    # Recording loop (prosthesis movement, EMG just records in background)
    all_records = []
    headers = None

    for itr in range(1, args.record_iterations + 1):
        print(f"Starting iteration {itr}/{args.record_iterations}...")
        arm.resetRecording()
        arm.recording = True
        arm.mainControlLoop(posDes=smooth_traj, period=10, emg=None)
        arm.recording = False

        # Store angle data
        raw_data = arm.recordedData
        if headers is None:
            headers = raw_data[0]  # first row is header names
        data_rows = raw_data[1:]
        all_records.extend(data_rows)

    # Stop raw EMG (and video) recorder
    if not args.no_emg:
        stop_event.set()
        raw_thread.join()
        if video_thread:
            video_thread.join()

    # Save all data: EMG = [N,16], timestamps = [N,], video timestamps if recorded
    if raw_history:
        np.save(join(base_dir, "raw_emg.npy"), np.vstack(raw_history))
        np.save(join(base_dir, "raw_timestamps.npy"), np.array(raw_timestamps))
        print(f"Saved raw_emg.npy with {len(raw_history)} samples.")
    if video_timestamps:
        np.save(join(base_dir, "video_timestamps.npy"), np.array(video_timestamps))
        print(f"Saved video_timestamps.npy with {len(video_timestamps)} frames.")

    # Save angle data
    if all_records:
        rec = np.array(all_records, dtype=float)
        ts = rec[:, 0]
        ts -= ts[0]  # normalize timestamps
        np.save(join(base_dir, "angles.npy"), rec)
        np.save(join(base_dir, "angle_timestamps.npy"), ts)
        with open(join(base_dir, "angles_header.txt"), "w") as f:
            f.write(",".join(headers))
        print(f"Saved angles.npy with {len(rec)} frames.")

    arm.close()

if __name__ == "__main__":
    main()

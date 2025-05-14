import argparse
import os
import time
import threading
import numpy as np
from os.path import join
from helpers.hand_poses import hand_poses
from psyonicHand import psyonicArm
from scipy.interpolate import CubicSpline
from helpers.EMGClass import EMG

def start_raw_emg_recorder():
    """
    Start background thread to record raw EMG at full board rate.
    Returns: stop_event, thread, raw_history, raw_timestamps
    """
    emg = EMG()  # connect directly to ADS1299 via EMGClass
    emg.startCommunication()

    raw_history = []
    raw_timestamps = []
    stop_event = threading.Event()

    def capture_loop():
        # wait for first EMG packet
        while emg.OS_time is None:
            time.sleep(1e-3)
        last_time = emg.OS_time
        # capture until stop_event set
        while not stop_event.is_set():
            t = emg.OS_time
            if t is not None and t > last_time:
                raw_history.append(list(emg.rawEMG))
                raw_timestamps.append(t)
                last_time = t
            time.sleep(1e-4)
        emg.exitEvent.set()

    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()
    return stop_event, thread, raw_history, raw_timestamps

def main():
    parser = argparse.ArgumentParser(
        description="Record synchronized EMG + prosthetic hand pose data"
    )
    parser.add_argument(
        "--person_id", "-p",
        required=True,
        help="Person ID (folder under data/)"
    )
    parser.add_argument(
        "--movement", "-m",
        required=True,
        help="Movement name (subfolder under recordings/)"
    )
    parser.add_argument(
        "--out_root", "-o",
        default="data",
        help="Root data directory (default: ./data)"
    )
    parser.add_argument(
        "--no_emg",
        action="store_true",
        help="Disable EMG capture for testing without hardware"
    )
    parser.add_argument(
        "--no_prosthesis",
        action="store_true",
        help="Disable prosthetic arm control; EMG-only recording"
    )
    parser.add_argument(
        "--hand_side", "-s",
        choices=["left", "right"],
        default="left",
        help="Side of the prosthetic hand (default: left)"
    )
    parser.add_argument(
        "--sync_iterations", 
        type=int,
        default=1,
        help="Number of warm-up sync iterations (default: 7)"
    )
    parser.add_argument(
        "--record_iterations", "-r",
        type=int,
        default=1,
        help="Number of recording iterations to perform (default: 25)"
    )
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

    # Start raw EMG recorder in background
    if not args.no_emg:
        stop_event, raw_thread, raw_history, raw_timestamps = start_raw_emg_recorder()
    else:
        stop_event = raw_thread = None
        raw_history = []
        raw_timestamps = []

    # If user only wants raw EMG, skip everything else
    if args.no_prosthesis:
        # Record raw EMG for this many seconds (make it its own CLI flag if you like)
        raw_seconds = 20.0  
        print(f"Raw-only mode: recording raw EMG for {raw_seconds:.1f} seconds…")
        try:
            time.sleep(raw_seconds)
        except KeyboardInterrupt:
            print("Interrupted early by user")
        # Stop the background recorder
        stop_event.set()
        raw_thread.join()
        # Save raw EMG
        if raw_history:
            np.save(join(base_dir, "raw_emg.npy"), np.vstack(raw_history))
            np.save(join(base_dir, "raw_timestamps.npy"), np.array(raw_timestamps))
            print(f"Saved raw_emg.npy with {len(raw_history)} samples.")
        else:
            print("No raw data captured.")
        print("Done.")
        return

    # Initialize prosthetic control EMG (filtered) for control loop
    if not args.no_emg:
        emg = EMG(
            socketAddr='tcp://127.0.0.1:1236',
            usedChannels=[0, 1, 2, 4, 5, 8, 10, 11]
        )
        emg.startCommunication()
        print("Filtered EMG subscriber connected for control.")
    else:
        emg = None
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
    neutral = np.array([2,2,2,2,2,-2])
    if isinstance(pose[0], (list, np.ndarray)):
        key_poses = [neutral] + [np.array(p) for p in pose] + [neutral]
    else:
        key_poses = [neutral, np.array(pose), neutral]
    total_dur = 2.0
    steps = 120
    times = np.linspace(0, total_dur, len(key_poses))
    interp = np.linspace(0, total_dur, steps)
    traj = CubicSpline(times, np.vstack(key_poses), axis=0)(interp)
    # enforce constant speed
    diffs = np.diff(traj, axis=0)
    dist = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0], np.cumsum(dist)])
    s_u = np.linspace(0, s[-1], len(s))
    const_traj = np.zeros_like(traj)
    idx = np.clip(np.searchsorted(s, s_u)-1, 0, len(s)-2)
    for i, su in enumerate(s_u):
        i0 = idx[i]
        ds = s[i0+1] - s[i0]
        α = (su - s[i0]) / ds if ds>0 else 0
        const_traj[i] = traj[i0]*(1-α) + traj[i0+1]*α
    smooth_traj = const_traj

    # Warm-up sync
    for _ in range(args.sync_iterations):
        arm.mainControlLoop(posDes=smooth_traj, period=10, emg=emg)

    # Recording loop
    all_records = []
    headers = None
    joint_names = ['index','middle','ring','pinky','thumbFlex','thumbRot']
    angle_cols = [f"{j}_Pos" for j in joint_names]

    for itr in range(1, args.record_iterations+1):
        print(f"Starting iteration {itr}/{args.record_iterations}...")
        arm.resetRecording()
        arm.recording = True
        arm.mainControlLoop(posDes=smooth_traj, period=10, emg=emg)
        arm.recording = False

        raw_data = arm.recordedData
        if headers is None:
            headers = raw_data[0]
            angle_idxs = [headers.index(c) for c in angle_cols]
        data_rows = raw_data[1:]
        all_records.extend(data_rows)

    # Stop raw EMG recorder
    if not args.no_emg:
        stop_event.set()
        raw_thread.join()

    # Save EMG + angles
    rec = np.array(all_records, dtype=float)
    ts = rec[:,0]
    emg_data = rec[:,1:1+len(emg.usedChannels)] if emg else np.empty((0,0))
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
    if emg:
        emg.exitEvent.set()

if __name__ == "__main__":
    main()

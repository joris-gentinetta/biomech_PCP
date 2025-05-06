import argparse
import os
import time
import numpy as np
from helpers.hand_poses import hand_poses
from psyonicHand import psyonicArm
from scipy.interpolate import CubicSpline
from EMGClass import EMG
from s0_emgInterface import EMGStreamer
import inspect


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
        "--hand_side", "-s",
        choices=["left", "right"],
        default="left",
        help="Side of the prosthetic hand (default: left)"
    )
    parser.add_argument(
        "--sync_iterations", 
        type=int,
        default=7,
        help="Number of warm-up sync iterations (default: 7)"
    )
    parser.add_argument(
        "--record_iterations", "-r",
        type=int,
        default=25,
        help="Number of recording iterations to perform (default: 10)"
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

    # Initialize EMG
    if not args.no_emg:
        # print("Starting EMGStreamer (COM6 → ZMQ)…")
        # print("Loading EMGClass from:", inspect.getsourcefile(EMG))
        # streamer = EMGStreamer(
        #     socketAddr='tcp://127.0.0.1:1236',
        #     port='COM6',
        #     baudrate=921600
        # )
        # streamer.startCommunication()
        # # give the bridge a moment
        # time.sleep(0.05)

        print("Connecting EMGClass subscriber…")
        emg = EMG(
            socketAddr='tcp://127.0.0.1:1236',
            usedChannels=[0, 1, 2, 4, 5, 8, 10, 11]
        )
        emg.startCommunication()
        print("EMG board connected.")
    else:
        emg = None
        print("EMG capture disabled; running in no-EMG mode.")

    # Initialize the prosthetic arm
    arm = psyonicArm(hand=args.hand_side)
    arm.initSensors()
    arm.startComms()

    # Validate movement
    pose_name = args.movement
    if pose_name not in hand_poses:
        print(f"Unknown movement '{pose_name}'. Available: {list(hand_poses.keys())}")
        arm.close()
        return
    pose = hand_poses[pose_name]

    # Build key poses (insert neutral at start/end if needed)
    neutral_pose = np.array([2, 2, 2, 2, 2, -2])
    if pose_name == "indexFlDigitsEx":
        if isinstance(pose, list) and len(pose) == 2:
            pos1 = np.array(pose[0])
            pos2 = np.array(pose[1])
            key_poses = [pos1, pos2]
        else:
            raise ValueError("'indexFlDigitsEx' must have exactly two key poses.")
    else:
        if isinstance(pose[0], (list, np.ndarray)):
            key_poses = [neutral_pose] + [np.array(p) for p in pose] + [neutral_pose]
        else:
            key_poses = [neutral_pose, np.array(pose), neutral_pose]

    # Trajectory parameters
    total_duration = 2.0  # seconds per cycle
    interp_steps = 120
    if pose_name == "indexFlDigitsEx":
        half = total_duration / 2
        t_half = np.linspace(0, half, interp_steps // 2)
        traj1 = CubicSpline([0, half], np.vstack([key_poses[0], key_poses[1]]), axis=0)(t_half)
        traj2 = CubicSpline([0, half], np.vstack([key_poses[1], key_poses[0]]), axis=0)(t_half)
        smooth_traj = np.vstack([traj1, traj2])
    else:
        key_times = np.linspace(0, total_duration, len(key_poses))
        t_interp = np.linspace(0, total_duration, interp_steps)
        poses_arr = np.vstack(key_poses)
        smooth_traj = CubicSpline(key_times, poses_arr, axis=0)(t_interp)

    # 1) compute cumulative distance along the curve
    diffs = np.diff(smooth_traj, axis=0)                  # (N-1,6)
    step_dists = np.linalg.norm(diffs, axis=1)            # (N-1,)
    s = np.concatenate([[0], np.cumsum(step_dists)])      # (N,); s[-1] is total length

    # 2) sample N points at *uniform* arc‐length
    N = smooth_traj.shape[0]
    s_uniform = np.linspace(0, s[-1], N)

    # 3) for each target s, find which segment it lies in & linearly interp
    traj_const_speed = np.zeros_like(smooth_traj)         # (N,6)
    idx = np.searchsorted(s, s_uniform) - 1
    idx = np.clip(idx, 0, N-2)

    for i, su in enumerate(s_uniform):
        i0 = idx[i]
        ds = s[i0+1] - s[i0]
        α  = (su - s[i0]) / ds if ds>0 else 0
        traj_const_speed[i] = smooth_traj[i0] * (1-α) + smooth_traj[i0+1] * α

    # now overwrite your old trajectory
    smooth_traj = traj_const_speed


    arm.Hz = 10
    arm.loopRate = 7

    # Warm-up sync iterations
    print(f"Running {args.sync_iterations} sync iterations (no recording)...")
    for i in range(args.sync_iterations):
        arm.mainControlLoop(posDes=smooth_traj, period=10, emg=emg) 

    all_records = []
    headers = None
    joint_names = ['index','middle','ring','pinky','thumbFlex','thumbRot']
    angle_cols = [f"{j}_Pos" for j in joint_names]

    overall_start = time.time()

    for itr in range(1, args.record_iterations + 1):
        print(f"Starting recording iteration {itr} of {args.record_iterations}...")
        arm.resetRecording()
        arm.recording = True
        start = time.time()
        arm.mainControlLoop(posDes=smooth_traj, period=10, emg=emg)
        arm.recording = False
        duration = time.time() - start
        
        raw = arm.recordedData
        if not raw or not isinstance(raw[0][0], str):
            raise RuntimeError("Expected recorded Data [0] to be header row")
        
        if headers is None:
            headers = raw[0]
            angle_idxs = [headers.index(col) for col in angle_cols]
            print("Angle columns:", [headers[i] for i in angle_idxs])

        data_rows = raw[1:]
        all_records.extend(data_rows)

        overall_duration = time.time() - overall_start

        print(f"Iteration {itr} completed in {duration:.2f} s.")
        print(f"Duration: {overall_duration}")

    # Convert all records to float array
    rec = np.array(all_records, dtype=float)
    timestamps = rec[:, 0]
    emg_data   = rec[:, 1:9]
    angles     = rec[:, angle_idxs]

    # Save to .npy files
    np.save(os.path.join(base_dir, "emg.npy"), emg_data)
    np.save(os.path.join(base_dir, "emg_timestamps.npy"), timestamps)
    np.save(os.path.join(base_dir, "angles.npy"), angles)
    np.save(os.path.join(base_dir, "angle_timestamps.npy"), timestamps)

    print(f"Saved EMG and angle data to {base_dir}")

    # Cleanup
    arm.close()
    if emg:
        emg.exitEvent.set()

if __name__ == "__main__":
    main()

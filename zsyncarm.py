#!/usr/bin/env python3
"""
Standalone script to synchronously record Psyonic hand positions and EMG samples.
Generates a simple sine-wave trajectory on the index finger and logs one EMG sample per hand loop step.
Trajectory is clipped to the hand's defined range of motion.
"""
import os
import time
import numpy as np
from psyonicHand import psyonicArm
from helpers.EMGClass import EMG


def main():
    # Directory to save the synced data
    base_dir = "synced_test"
    os.makedirs(base_dir, exist_ok=True)

    # Initialize the prosthetic hand
    print("Initializing Psyonic Arm...")
    arm = psyonicArm(hand='right')  # or 'left'
    arm.initSensors()
    arm.startComms()
    time.sleep(0.2)  # allow comm thread to start

    # Initialize EMG acquisition
    print("Initializing EMG...")
    emg = EMG()
    emg.readEMG()  # prime EMG OS_time

    # Build a simple sine trajectory for the index finger
    start_pos = np.array(arm.getCurPos(), dtype=float)
    num_steps = 300  # number of samples (e.g., 5 seconds @ 60Hz)
    t = np.linspace(0, 2 * np.pi, num_steps)
    traj = np.tile(start_pos, (num_steps, 1)).astype(float)
    # Move only the index joint ±30° around its start
    traj[:, 0] = traj[:, 0] + 30.0 * np.sin(t)

    # Clip trajectory to the hand's range of motion
    for idx, jname in enumerate(arm.jointNames):
        low, high = arm.jointRoM[jname]
        traj[:, idx] = np.clip(traj[:, idx], low, high)
    print(f"Generated and clipped trajectory with {num_steps} steps.")

    # Reset and enable recording in the arm
    arm.resetRecording()
    arm.recording = True

    # Run the synchronous loop: one EMG sample per hand command
    print("Starting synchronous recording...")
    arm.mainControlLoop(emg=emg, posDes=traj, period=1)
    arm.recording = False
    print("Recording finished.")

    # Retrieve and save the logged data
    raw = arm.recordedData
    headers = raw[0]
    data = np.array(raw[1:], dtype=float)

    # Save to .npy and header file
    np.save(os.path.join(base_dir, 'synced_data.npy'), data)
    with open(os.path.join(base_dir, 'headers.txt'), 'w') as f:
        f.write(','.join(headers))
    print(f"Saved {data.shape[0]} samples to '{base_dir}/synced_data.npy'.")

    # Clean up EMG
    try:
        if hasattr(emg, 'exitEvent'):
            emg.exitEvent.set()
        if hasattr(emg, 'shutdown'):
            emg.shutdown()
    except Exception:
        pass

    # Clean up arm communication
    arm.close()
    print("Done.")


if __name__ == '__main__':
    main()

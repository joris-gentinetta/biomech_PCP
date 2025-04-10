import sys
import os
import time
import numpy as np
from helpers.hand_poses import hand_poses
from psyonicHand import psyonicArm
from scipy.interpolate import CubicSpline
from EMGClass import EMG  # Ensure this module is in the PYTHONPATH

# ---------------------------
# Ask the user for the output directory.
output_dir = input("Enter the output directory for recorded data: ")
if not os.path.isdir(output_dir):
    print("Directory does not exist. Creating it.")
    os.makedirs(output_dir)

# ---------------------------
# Connect to the EMG board.
# print("Connecting to EMG board...")
# emg = EMG(usedChannels=[0, 1, 2, 4, 5, 8, 10, 11])
# emg.startCommunication()
# print("EMG board connected.")

# ---------------------------
# Initialize the hand (Ability Hand)
arm = psyonicArm(hand='left')
arm.initSensors()
arm.startComms()

# Define the neutral hand pose
neutral_pose = np.array([5, 5, 5, 5, 5, -5])

# ---------------------------
# Ask the user which hand pose to run.
pose_name = input(f"Enter pose name {list(hand_poses.keys())}: ")
if pose_name not in hand_poses:
    print(f"Pose '{pose_name}' not found.")
    arm.close()
    sys.exit()

pose = hand_poses[pose_name]

# ---------------------------
# Build the key poses for the desired movement.
# Special case: "indexFlDigitsEx" must contain exactly two positions.
if pose_name == "indexFlDigitsEx":
    if isinstance(pose, list) and len(pose) == 2:
        pos1 = np.array(pose[0])
        pos2 = np.array(pose[1])
        key_poses = [pos1, pos2]
    else:
        print("Error: The pose 'indexFlDigitsEx' must contain exactly two positions.")
        arm.close()
        sys.exit()
else:
    # For other poses, if sub-steps are provided, wrap with neutral pose at start and end.
    if isinstance(pose[0], (list, np.ndarray)):
        key_poses = [neutral_pose] + pose + [neutral_pose]
    else:
        key_poses = [neutral_pose, pose, neutral_pose]

# ---------------------------
# Movement parameters
total_duration = 2.0  # seconds for one full trajectory cycle
sync_iterations = 7   # Number of iterations to run without recording (to allow user syncing)
record_iterations = 10 # We'll record on one iteration (the 8th)

# Initial command to send the arm to a neutral position
arm.mainControlLoop(posDes=neutral_pose, period=0.08)
time.sleep(1)

# ---------------------------
# Generate a smooth trajectory using cubic spline interpolation.
interp_steps = 120
if pose_name == "indexFlDigitsEx":
    # Generate a trajectory that goes from pos1 to pos2 (half_cycle) and then back to pos1.
    half_duration = total_duration / 2
    times_half = np.linspace(0, half_duration, interp_steps // 2)
    # Interpolate from first key pose to second.
    traj_to = CubicSpline([0, half_duration],
                          np.vstack([key_poses[0], key_poses[1]]),
                          axis=0)(times_half)
    # Interpolate back from second key pose to first.
    traj_back = CubicSpline([0, half_duration],
                            np.vstack([key_poses[1], key_poses[0]]),
                            axis=0)(times_half)
    smooth_trajectory = np.vstack([traj_to, traj_back])
else:
    # For normal poses, define key times and interpolate.
    key_times = np.linspace(0, total_duration, len(key_poses))
    interp_times = np.linspace(0, total_duration, interp_steps)
    key_poses_array = np.vstack(key_poses)
    smooth_trajectory = CubicSpline(key_times, key_poses_array, axis=0)(interp_times)

# ---------------------------
# Run the pose for sync iterations (without recording) so the user can acclimate.
print("\nSynchronizing with the hand's movement. Running 7 iterations (no recording).")
for i in range(sync_iterations):
    print(f"Sync iteration {i+1} of {sync_iterations}")
    arm.mainControlLoop(posDes=smooth_trajectory, period=0.06)
    time.sleep(0.1)

# ---------------------------
# Now begin recording.
print("\nStarting recording on iteration 8.")
arm.resetRecording()   # Clear any previous recordings in the arm's internal log.
arm.recording = True

# For this example, we record one full iteration while the arm executes the smooth trajectory.
print("Recording iteration...")
recording_start_time = time.time()
# We pass the EMG object as well so that arm.addLogEntry(emg) can grab the EMG signal.
# arm.mainControlLoop(posDes=smooth_trajectory, period=0.06, emg=emg)
arm.mainControlLoop(posDes=smooth_trajectory, period=0.06)
print(f"Recording iteration completed in {time.time() - recording_start_time:.2f} seconds.")

arm.recording = False

# ---------------------------
# Ask the user for a filename to save the recorded data.
filename = pose_name
filepath = os.path.join(output_dir, filename + ".csv")

# Save the recorded data.
# arm.recordedData is assumed to be a list of lists (each row is a log entry with timestamp, EMG, joint positions, etc.).
recorded_data = np.array(arm.recordedData)
np.savetxt(filepath, recorded_data, delimiter='\t', fmt='%s')
print(f"Recorded data saved to: {filepath}")

# ---------------------------
# Clean up communications.
arm.close()
# emg.exitEvent.set()
print("Arm and EMG communications closed.")

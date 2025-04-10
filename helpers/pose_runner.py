import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
from hand_poses import hand_poses
from psyonicHand import psyonicArm
from scipy.interpolate import CubicSpline

# Initialize arm
arm = psyonicArm(hand='left')
arm.initSensors()
arm.startComms()

# Neutral hand pose
neutral_pose = np.array([5, 5, 5, 5, 5, -5])

# Select pose
pose_name = input(f"Enter pose name {list(hand_poses.keys())}: ")
if pose_name not in hand_poses:
    print(f"Pose '{pose_name}' not found.")
    arm.close()
    exit()

pose = hand_poses[pose_name]

# Special case of IndexFlDigitsEx
if pose_name == "indexFlDigitsEx":
    if isinstance(pose, list) and len(pose) == 2:
        pos1 = np.array(pose[0])
        pos2 = np.array(pose[1])
        key_poses = [pos1, pos2]
    else:
        print("Error: the pose must contain exactly two positions")
        arm.close()
        exit
        
else: 
    # Handle poses with sub-steps (collision-safe) or single-step poses
    if isinstance(pose[0], (list, np.ndarray)):
        key_poses = [neutral_pose] + pose + [neutral_pose]
    else:
        key_poses = [neutral_pose, pose, neutral_pose]

# Movement parameters
total_duration = 2  # total duration for one trajectory (seconds)
iterations = int(input("Enter number of iterations: "))

# Initial pause command before beginning trajectory execution
arm.mainControlLoop(posDes=neutral_pose, period=0.08)
time.sleep(1)

if pose_name == "indexFlDigitsEx":
    # Generate a smooth trajectory using cubic spline interpolation
    interp_steps = 120  # higher number yields smoother trajectory; adjust as needed
    half_duration = total_duration /2
    times_half = np.linspace(0, half_duration, interp_steps // 2)
    traj_to = CubicSpline([0, half_duration], np.vstack([key_poses[0], key_poses[1]]), axis=0)(times_half)
    traj_back = CubicSpline([0, half_duration], np.vstack([key_poses[1], key_poses[0]]), axis=0)(times_half)
    smooth_trajectory = np.vstack([traj_to, traj_back])
else:
    interp_steps = 120
    key_times = np.linspace(0, total_duration, len(key_poses))
    interp_times = np.linspace(0, total_duration, interp_steps)
    # Convert list of key poses into a 2D array (rows: key poses, columns: joint values)
    key_poses_array = np.vstack(key_poses)
    smooth_trajectory = CubicSpline(key_times, key_poses_array, axis=0)(interp_times)

# Execute the smooth trajectory for the specified number of iterations
try:
    for itr in range(iterations):
        print(f"Iteration {itr + 1} of {iterations}")
        iteration_start_time = time.time()

        # Command the arm with the smooth interpolated trajectory
        arm.mainControlLoop(posDes=smooth_trajectory, period=0.06)

        print(f"Iteration {itr + 1} completed in {time.time() - iteration_start_time:.2f}s")
        # Pause for 1 second between iterations
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Execution interrupted by user.")

finally:
    arm.close()
    print("Arm communication closed.")

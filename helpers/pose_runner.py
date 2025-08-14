import sys
import os
import time
import numpy as np
from scipy.interpolate import CubicSpline

# Make sure your path is set up for the local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hand_poses import hand_poses
from psyonicHand import psyonicArm

# ----------- PARAMETERS -----------
neutral_pose = np.array([2, 2, 2, 2, 2, -2])
move_duration = 2.0      # seconds to go from neutral to target or back
hold_duration = 0.7      # hold at target and neutral, in seconds
steps_per_sec = 300      # controls smoothness (also set arm period accordingly)

# ----------- USER INPUT -----------
pose_name = input(f"Enter pose name {list(hand_poses.keys())}: ")
if pose_name not in hand_poses:
    print(f"Pose '{pose_name}' not found.")
    exit()

iterations = int(input("Enter number of iterations: "))

# ----------- LOAD POSE -----------
pose = hand_poses[pose_name]
if isinstance(pose, list) and isinstance(pose[0], (list, np.ndarray)):
    # Use the most "open" or "closed" (max deviation from neutral)
    pose_array = np.array(pose)
    target_pose = pose_array[np.argmax(np.linalg.norm(pose_array - neutral_pose, axis=1))]
else:
    target_pose = np.array(pose)

# ----------- TRAJECTORY CONSTRUCTION -----------
n_steps = int(move_duration * steps_per_sec)
hold_steps = int(hold_duration * steps_per_sec)

# Smooth trajectory neutral -> target
traj_to = CubicSpline([0, move_duration], np.vstack([neutral_pose, target_pose]), axis=0)(
    np.linspace(0, move_duration, n_steps)
)
# Smooth trajectory target -> neutral
traj_back = CubicSpline([0, move_duration], np.vstack([target_pose, neutral_pose]), axis=0)(
    np.linspace(0, move_duration, n_steps)
)
# Holds
hold_neutral = np.tile(neutral_pose, (hold_steps, 1))
hold_target = np.tile(target_pose, (hold_steps, 1))

# Full cycle: hold at neutral → move to target → hold at target → move to neutral → hold at neutral (optional, for annotation)
full_traj = np.vstack([
    hold_neutral,
    traj_to,
    hold_target,
    traj_back,
    hold_neutral
])

# ----------- ARM EXECUTION -----------
arm = psyonicArm(hand='left')
arm.initSensors()
arm.startComms()
time.sleep(0.5)  # let the hardware settle

try:
    for itr in range(iterations):
        print(f"Iteration {itr + 1} of {iterations}")
        start = time.time()
        arm.mainControlLoop(posDes=full_traj, period=5)
        print(f"Iteration {itr + 1} completed in {time.time() - start:.2f}s")
        time.sleep(0.3)  # short pause between cycles

except KeyboardInterrupt:
    print("Execution interrupted by user.")

finally:
    arm.close()
    print("Arm communication closed.")


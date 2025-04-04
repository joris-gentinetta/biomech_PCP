import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
from hand_poses import hand_poses
from psyonicHand import psyonicArm

# Initialize the arm
arm = psyonicArm(hand='left')
arm.initSensors()
arm.startComms()


# Select pose
pose_name = input(f"Enter pose name {list(hand_poses.keys())}: ")
if pose_name not in hand_poses:
    print(f"Pose '{pose_name}' not found.")
    exit()

neutral_pose = np.array([5, 5, 5, 5, 5, -5]) 
target_pose = np.array(hand_poses[pose_name])
iterations = 4
period = 0.07
delay = 0.5

print(f"Executing pose '{pose_name}' for {iterations} iterations...")

try:
    for i in range(iterations):
        print(f"Iteration {i+1}")
        arm.mainControlLoop(posDes=target_pose, period=period)
        time.sleep(delay)
        arm.mainControlLoop(posDes=neutral_pose, period=period)
        time.sleep(delay)
except KeyboardInterrupt:
    pass

arm.close()

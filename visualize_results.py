import pandas as pd
import pybullet as p
import cv2
from time import sleep
import math
multiplier = 1.05851325
side = 'Left'

# Start PyBullet in GUI mode to visualize the robot (use p.DIRECT for no GUI)
physicsClient = p.connect(p.GUI)
def get_joints_of_type(body_id, joint_type):
    joint_indices = []
    num_joints = p.getNumJoints(body_id)
    for i in range(num_joints):
        info = p.getJointInfo(body_id, i)
        if info[2] == joint_type:
            joint_indices.append(i)
    return joint_indices

p.setGravity(0, 0, -10)

handStartPos = [0,0,0]
handStartOrientation = p.getQuaternionFromEuler([0,0,0])
handId = p.loadURDF("URDF/ability_hand_left_large.urdf", handStartPos, handStartOrientation,
                     flags=p.URDF_USE_SELF_COLLISION, useFixedBase=True)

joints_of_type_0 = get_joints_of_type(handId, 0)

# Set the camera parameters
camera_distance = 1.34  # Closer distance makes the "zoom" effect
camera_yaw = 223  # Adjust as needed for best angle
camera_pitch = -25  # Adjust as needed
camera_target_position = [0.59, -0.65, -0.38]  # Focus on the center of your model or a specific part

# Reset the camera with the specified parameters
p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

joint_ids = {'index': (1, 2, 3), 'middle': (4, 5, 6), 'ring': (7, 8, 9), 'pinky': (10, 11, 12), 'thumb': (13, 14, 15)}

def move_finger(finger, angle):
    angle = math.radians(angle)
    id0, id1 = joint_ids[finger][0], joint_ids[finger][1]
    p.setJointMotorControl2(handId, id0, p.POSITION_CONTROL, targetPosition=angle)
    p.setJointMotorControl2(handId, id1, p.POSITION_CONTROL, targetPosition=angle * multiplier)


# Load the video
video_path = '/Users/jg/projects/biomech/DataGen/data/joris/camera_test/experiments/1/visualization_corrected.mp4'
cap = cv2.VideoCapture(video_path)
angles = pd.read_parquet('/Users/jg/projects/biomech/DataGen/data/joris/camera_test/experiments/1/angles.parquet')
for i in range(angles.shape[0]):
    move_finger('index', angles.loc[i, (side, 'indexAng')])
    move_finger('middle', angles.loc[i, (side, 'midAng')])
    move_finger('ring', angles.loc[i, (side, 'ringAng')])
    move_finger('pinky', angles.loc[i, (side, 'pinkyAng')])

    #thumb:
    angle = angles.loc[i, (side, 'thumbInPlaneAng')]
    p.setJointMotorControl2(handId, joint_ids['thumb'][0], p.POSITION_CONTROL, targetPosition=angle)
    angle = angles.loc[i, (side, 'thumbOutPlaneAng')]
    p.setJointMotorControl2(handId, joint_ids['thumb'][1], p.POSITION_CONTROL, targetPosition=angle)

    ret, frame = cap.read()
    cv2.imshow('Frame', frame)
    if i == 0:
        cv2.waitKey(0)

    for i in range(100):
        p.stepSimulation()

    sleep(1/cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# while True:
#     p.stepSimulation()
#     time.sleep(1./240.)
cap.release()
cv2.destroyAllWindows()

import os.path

import pandas as pd
import pybullet as p
import cv2
from time import sleep
import argparse
from os.path import join
from helpers.utils import AnglesHelper
multiplier = 1.05851325
offset = 0.72349796


def get_joints_of_type(body_id, joint_type):
    joint_indices = []
    num_joints = p.getNumJoints(body_id)
    for i in range(num_joints):
        info = p.getJointInfo(body_id, i)
        if info[2] == joint_type:
            joint_indices.append(i)
    return joint_indices



def move_finger(finger, angle):
    id0, id1 = joint_ids[finger][0], joint_ids[finger][1]
    p.setJointMotorControl2(handId, id0, p.POSITION_CONTROL, targetPosition=angle)
    p.setJointMotorControl2(handId, id1, p.POSITION_CONTROL, targetPosition=angle * multiplier + offset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture a video.')
    parser.add_argument('--data_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--intact_hand', type=str, default=None, help='Intact hand')
    parser.add_argument('--model', type=str, default=None, help='Model name')
    args = parser.parse_args()

    cap = cv2.VideoCapture(join(args.data_dir, 'experiments', args.experiment_name, 'visualization_corrected.mp4'))
    angles = pd.read_parquet(join(args.data_dir, 'experiments', args.experiment_name, 'cropped_smooth_angles.parquet'))

    if args.model:
        if not os.path.exists(join(args.data_dir, 'experiments', args.experiment_name, 'visualization_test.mp4')):
            ##########################################
            # make a new video with all frames after len(angles)//5 * 4:
            import cv2
            from os.path import join

            # Define the codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
            out = cv2.VideoWriter(join(args.data_dir, 'experiments', args.experiment_name, 'visualization_test.mp4'), fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            for _ in range(len(angles)//5 * 4):
                ret, frame = cap.read()

            # Read and write the remaining frames to the new video
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                else:
                    break

            # Release everything when job is finished
            cap.release()
            out.release()
            ##########################################
        cap = cv2.VideoCapture(join(args.data_dir, 'experiments', args.experiment_name, 'visualization_test.mp4'))
        angles = pd.read_parquet(join(args.data_dir, 'experiments', args.experiment_name, f'pred_angles_{args.model}.parquet')) #todo
        # angles.index = range(len(angles))
        # angles = angles.loc[len(angles)//5 * 4:].copy()
    ######

    angles.index = range(len(angles))


    # angles.loc[:, :] = 0  # todo
    # anglesHelper = AnglesHelper()
    # angles = anglesHelper.apply_gaussian_smoothing(angles, sigma=1.5, radius=2) #todo
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -10)

    handStartPos = [0, 0, 0]
    handStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    urdf_path = "URDF/ability_hand_left_large.urdf" if args.intact_hand == 'Left' else "URDF/ability_hand_right_large.urdf"
    handId = p.loadURDF(urdf_path, handStartPos, handStartOrientation,
                        flags=p.URDF_USE_SELF_COLLISION, useFixedBase=True)

    # visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 1])
    # point_id = p.createMultiBody(baseMass=0,
    #                              baseInertialFramePosition=[0, 0, 0],
    #                              baseVisualShapeIndex=visual_shape_id,
    #                              basePosition=[0, 0, 1])

    joints_of_type_0 = get_joints_of_type(handId, 0)

    # Set the camera parameters
    camera_distance = 1.34  # Closer distance makes the "zoom" effect
    camera_yaw = 223  # Adjust as needed for best angle
    camera_pitch = -25  # Adjust as needed
    camera_target_position = [0.59, -0.65, -0.38]  # Focus on the center of your model or a specific part
    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

    joint_ids = {'index': (1, 2, 3), 'middle': (4, 5, 6), 'ring': (7, 8, 9), 'pinky': (10, 11, 12),
                 'thumb': (13, 14, 15)}

    for i in range(angles.shape[0]):
        move_finger('index', angles.loc[i, (args.intact_hand, 'indexAng')])
        move_finger('middle', angles.loc[i, (args.intact_hand, 'midAng')])
        move_finger('ring', angles.loc[i, (args.intact_hand, 'ringAng')])
        move_finger('pinky', angles.loc[i, (args.intact_hand, 'pinkyAng')])

        # x = angles.loc[i, (args.intact_hand, 'thumb_x')]
        # y = angles.loc[i, (args.intact_hand, 'thumb_y')]
        # z = angles.loc[i, (args.intact_hand, 'thumb_z')]
        # p.resetBasePositionAndOrientation(point_id, [x, y, z], [0, 0, 0, 1])


        #thumb:
        angle = angles.loc[i, (args.intact_hand, 'thumbInPlaneAng')]
        p.setJointMotorControl2(handId, joint_ids['thumb'][0], p.POSITION_CONTROL, targetPosition=angle)
        angle = angles.loc[i, (args.intact_hand, 'thumbOutPlaneAng')]
        p.setJointMotorControl2(handId, joint_ids['thumb'][1], p.POSITION_CONTROL, targetPosition=angle)

        ret, frame = cap.read()
        cv2.imshow('Frame', frame)
        if i == 0:
            cv2.waitKey(0)

        for i in range(20):
            p.stepSimulation()

        # sleep(1/cap.get(cv2.CAP_PROP_FPS))

    cap.release()
    cv2.destroyAllWindows()

import pandas as pd
import pybullet as p
import cv2
import argparse
from os.path import join, exists
from time import sleep
import time
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



def move_finger(hand, finger, angle):
    id0, id1 = joint_ids[finger][0], joint_ids[finger][1]
    p.setJointMotorControl2(hand, id0, p.POSITION_CONTROL, targetPosition=angle)
    p.setJointMotorControl2(hand, id1, p.POSITION_CONTROL, targetPosition=angle * multiplier + offset)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture a video.')
    parser.add_argument('--data_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--intact_hand', type=str, default=None, help='Intact hand')
    parser.add_argument('--model_name', type=str, default=None, help='Name specified in the config file used to train the model')
    parser.add_argument('--video', action='store_true', help='Display the video')
    args = parser.parse_args()

    if args.video:
        cap = cv2.VideoCapture(join(args.data_dir, 'experiments', args.experiment_name, 'visualization_corrected.mp4'))
    target_angles = pd.read_parquet(join(args.data_dir, 'experiments', args.experiment_name, 'cropped_smooth_angles.parquet'))

    if args.model_name:
        if not exists(join(args.data_dir, 'experiments', args.experiment_name, 'visualization_test.mp4')) and args.video:

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
            out = cv2.VideoWriter(join(args.data_dir, 'experiments', args.experiment_name, 'visualization_test.mp4'), fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            start_frame = len(target_angles)//5 * 4
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                else:
                    break

            cap.release()
            out.release()
        cap = cv2.VideoCapture(join(args.data_dir, 'experiments', args.experiment_name, 'visualization_test.mp4'))

        pred_angles = pd.read_parquet(join(args.data_dir, 'experiments', args.experiment_name, f'pred_angles-{args.model_name}.parquet'))
        pred_angles.index = range(len(pred_angles))
        target_angles.index = range(len(target_angles))
        target_angles = target_angles.loc[len(target_angles)//5 * 4:].copy()
        target_angles.index = range(len(target_angles))

        # # # todo remove:
        # from helpers.utils import AnglesHelper
        # angles_helper = AnglesHelper()
        # target_angles = angles_helper.mirror_angles(target_angles, 'Right')
        # pred_angles = angles_helper.mirror_angles(pred_angles, 'Right')

    physicsClient = p.connect(p.GUI)
    # p.setRealTimeSimulation(0)
    # p.setTimeStep(1 / 10000)


    p.setGravity(0, 0, -9.81)

    handStartPos = [0, 0, 0]
    handStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    urdf_path = "URDF/ability_hand_left_large-wrist.urdf"  if args.intact_hand == 'Left' else "URDF/ability_hand_right_large-wrist.urdf"
    target_hand = p.loadURDF(urdf_path, handStartPos, handStartOrientation,
                        flags=p.URDF_USE_SELF_COLLISION, useFixedBase=True)

    for i in range(1, p.getNumJoints(target_hand)):
        p.changeVisualShape(target_hand, i, rgbaColor=[0, 1, 0, 0.7])
    if args.model_name:
        pred_hand = p.loadURDF(urdf_path, handStartPos, handStartOrientation,
                            flags=p.URDF_USE_SELF_COLLISION, useFixedBase=True)
        for i in range(1, p.getNumJoints(pred_hand)):
            p.changeVisualShape(pred_hand, i, rgbaColor=[1, 0, 0, 0.7])
    # visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 1])
    # point_id = p.createMultiBody(baseMass=0,
    #                              baseInertialFramePosition=[0, 0, 0],
    #                              baseVisualShapeIndex=visual_shape_id,
    #                              basePosition=[0, 0, 1])


    camera_distance = 1.34  # Closer distance makes the "zoom" effect
    camera_yaw = 223  # Adjust as needed for best angle
    camera_pitch = -25  # Adjust as needed
    camera_target_position = [0.59, -0.65, -0.3]  # Focus on the center of your model or a specific part
    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

    # joint_ids = {'index': (1, 2, 3), 'middle': (4, 5, 6), 'ring': (7, 8, 9), 'pinky': (10, 11, 12),
    #              'thumb': (13, 14, 15)}

    joint_ids = {
        'elbow': 0,
        'wrist_rotation': 1,
        'wrist_flexion': 2,
        'index': (4, 5, 6),
        'middle': (7, 8, 9),
        'ring': (10, 11, 12),
        'pinky': (13, 14, 15),
        'thumb': (16, 17, 18)
    }

    time.sleep(1) # want to wait for the GUI to load
    t = time.time()

    for i in range(target_angles.shape[0]):
        move_finger(target_hand, 'index', target_angles.loc[i, (args.intact_hand, 'indexAng')])
        move_finger(target_hand, 'middle', target_angles.loc[i, (args.intact_hand, 'midAng')])
        move_finger(target_hand, 'ring', target_angles.loc[i, (args.intact_hand, 'ringAng')])
        move_finger(target_hand, 'pinky', target_angles.loc[i, (args.intact_hand, 'pinkyAng')])

        # x = angles.loc[i, (args.intact_hand, 'thumb_x')]
        # y = angles.loc[i, (args.intact_hand, 'thumb_y')]
        # z = angles.loc[i, (args.intact_hand, 'thumb_z')]
        # p.resetBasePositionAndOrientation(point_id, [x, y, z], [0, 0, 0, 1])

        #thumb:
        angle = target_angles.loc[i, (args.intact_hand, 'thumbInPlaneAng')]
        p.setJointMotorControl2(target_hand, joint_ids['thumb'][0], p.POSITION_CONTROL, targetPosition=angle)
        angle = target_angles.loc[i, (args.intact_hand, 'thumbOutPlaneAng')]
        p.setJointMotorControl2(target_hand, joint_ids['thumb'][1], p.POSITION_CONTROL, targetPosition=angle)

        #wrist:
        angle = target_angles.loc[i, (args.intact_hand, 'wristRot')]
        p.setJointMotorControl2(target_hand, joint_ids['wrist_rotation'], p.POSITION_CONTROL, targetPosition=angle)
        angle = target_angles.loc[i, (args.intact_hand, 'wristFlex')]
        p.setJointMotorControl2(target_hand, joint_ids['wrist_flexion'], p.POSITION_CONTROL, targetPosition=angle)


        if args.model_name:
            move_finger(pred_hand, 'index', pred_angles.loc[i, (args.intact_hand, 'indexAng')])
            move_finger(pred_hand, 'middle', pred_angles.loc[i, (args.intact_hand, 'midAng')])
            move_finger(pred_hand, 'ring', pred_angles.loc[i, (args.intact_hand, 'ringAng')])
            move_finger(pred_hand, 'pinky', pred_angles.loc[i, (args.intact_hand, 'pinkyAng')])

            #thumb:
            angle = pred_angles.loc[i, (args.intact_hand, 'thumbInPlaneAng')]
            p.setJointMotorControl2(pred_hand, joint_ids['thumb'][0], p.POSITION_CONTROL, targetPosition=angle)
            angle = pred_angles.loc[i, (args.intact_hand, 'thumbOutPlaneAng')]
            p.setJointMotorControl2(pred_hand, joint_ids['thumb'][1], p.POSITION_CONTROL, targetPosition=angle)

            # wrist:
            angle = pred_angles.loc[i, (args.intact_hand, 'wristRot')]
            p.setJointMotorControl2(pred_hand, joint_ids['wrist_rotation'], p.POSITION_CONTROL, targetPosition=angle)
            angle = pred_angles.loc[i, (args.intact_hand, 'wristFlex')]
            p.setJointMotorControl2(pred_hand, joint_ids['wrist_flexion'], p.POSITION_CONTROL, targetPosition=angle)

        if args.video:

            ret, frame = cap.read()
            cv2.imshow('Frame', frame)
            if i == 0:
                cv2.waitKey(0)

        # for i in range(20):
        #     p.stepSimulation()


        while (1 / (time.time() - t)) > 20:
            print(i, 1 / (time.time() - t))
            p.stepSimulation()
        print(1 / (time.time() - t))
        t = time.time()

    if args.video:
        cap.release()
        cv2.destroyAllWindows()

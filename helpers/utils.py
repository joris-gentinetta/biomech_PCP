import os.path

import numpy as np
import math
import zmq
import pybullet as p
from tqdm import tqdm
import pandas as pd
pd.options.mode.copy_on_write = True

class Comms:
    def __init__(self, sendSocketAddr):
        self._sendSocketAddr = sendSocketAddr
        self._sendCTX = zmq.Context()
        self.sendSock = self._sendCTX.socket(zmq.PUB)
        self.sendSock.bind(self._sendSocketAddr)

    def __del__(self):
        try:
            self.sendSock.unbind(self._sendSocketAddr)
            self.sendSock.close()
            self._sendCTX.term()

        except Exception as e:
            print(f'__del__: Socket closing error {e}')


class AnglesHelper:
    def __init__(self):
        pass

    @staticmethod
    def angleBetweenVectors(v1, v2):

        angle = np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))

        return angle


    def calculateIndex(self, output_df, i, side):
        vec1 = output_df.loc[i, (side, 'INDEX_FINGER_PIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'INDEX_FINGER_MCP', ['x', 'y', 'z'])].values
        vec2 = output_df.loc[i, (side, 'INDEX_FINGER_MCP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'WRIST', ['x', 'y', 'z'])].values
        mcpAng = self.angleBetweenVectors(vec1, vec2)

        vec4 =  output_df.loc[i, (side, 'INDEX_FINGER_DIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'INDEX_FINGER_PIP', ['x', 'y', 'z'])].values
        pipAng = self.angleBetweenVectors(vec1, vec4)

        return max(pipAng, mcpAng)


    def calculateMiddle(self, output_df, i, side):
        vec1 = output_df.loc[i, (side, 'MIDDLE_FINGER_PIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'MIDDLE_FINGER_MCP', ['x', 'y', 'z'])].values
        vec2 = output_df.loc[i, (side, 'MIDDLE_FINGER_MCP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'WRIST', ['x', 'y', 'z'])].values
        mcpAng = self.angleBetweenVectors(vec1, vec2)

        vec4 = output_df.loc[i, (side, 'MIDDLE_FINGER_DIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'MIDDLE_FINGER_PIP', ['x', 'y', 'z'])].values
        pipAng = self.angleBetweenVectors(vec1, vec4)

        return max(pipAng, mcpAng)


    def calculateRing(self, output_df, i, side):
        vec1 = output_df.loc[i, (side, 'RING_FINGER_PIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'RING_FINGER_MCP', ['x', 'y', 'z'])].values
        vec2 = output_df.loc[i, (side, 'RING_FINGER_MCP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'WRIST', ['x', 'y', 'z'])].values
        mcpAng = self.angleBetweenVectors(vec1, vec2)

        vec4 = output_df.loc[i, (side, 'RING_FINGER_DIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'RING_FINGER_PIP', ['x', 'y', 'z'])].values
        pipAng = self.angleBetweenVectors(vec1, vec4)

        return max(pipAng, mcpAng)


    def calculatePinky(self, output_df, i, side):
        vec1 = output_df.loc[i, (side, 'PINKY_PIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'PINKY_MCP', ['x', 'y', 'z'])].values
        vec2 = output_df.loc[i, (side, 'PINKY_MCP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'WRIST', ['x', 'y', 'z'])].values
        mcpAng = self.angleBetweenVectors(vec1, vec2)

        vec4 = output_df.loc[i, (side, 'PINKY_DIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'PINKY_PIP', ['x', 'y', 'z'])].values
        pipAng = self.angleBetweenVectors(vec1, vec4)

        return max(pipAng, mcpAng)

    def get_mapping(self):
        if os.path.exists('mapping.npz'):
            mapping = np.load('mapping.npz')
            coords = mapping['coords']
            angles = mapping['angles']
            pinky_to_index_sim = mapping['pinky_to_index_sim'][0]
        else:
            coords = []
            angles = []

            physicsClient = p.connect(p.DIRECT)
            p.setGravity(0, 0, -10)
            handStartPos = [0, 0, 0]
            handStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
            handId = p.loadURDF("URDF/ability_hand_left_large.urdf", handStartPos, handStartOrientation,
                                flags=p.URDF_USE_SELF_COLLISION, useFixedBase=True)
            joint_limits_rot = p.getJointInfo(handId, 13)[8: 10]
            joint_limits_flex = p.getJointInfo(handId, 14)[8: 10]
            for rot in np.linspace(joint_limits_rot[0], joint_limits_rot[1], 100):
                for flex in np.linspace(joint_limits_flex[0], joint_limits_flex[1], 100):
                    p.resetJointState(handId, 13, rot)
                    p.resetJointState(handId, 14, flex)
                    p.stepSimulation()
                    # get coordinates of the thumb tip
                    thumb_tip = p.getLinkState(handId, 15)[0]
                    coords.append(thumb_tip)
                    angles.append((rot, flex))

            index_q1 = p.getLinkState(handId, 1)[0]
            pinky_q1 = p.getLinkState(handId, 10)[0]
            pinky_to_index_sim = index_q1[0] - pinky_q1[0]
            coords = np.array(coords)
            angles = np.array(angles)
            np.savez('mapping.npz', coords=coords, angles=angles, pinky_to_index_sim=np.array([pinky_to_index_sim]))

        return coords, angles, pinky_to_index_sim


    def getArmAngles(self, output_df, sides):
        angle_names = ['indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng', 'elbowAngle', 'wristRot',
                       'wristFlex']
        hand_names = ['Left', 'Right']

        columns = pd.MultiIndex.from_product([hand_names, angle_names])
        angles_df = pd.DataFrame(index=output_df.index, columns=columns)

        coords, angles, pinky_to_index_sim = self.get_mapping()

        print('\nCalculating angles...')
        for i in tqdm(angles_df.index):
            for side in sides:
                body_wrist = output_df.loc[i, (side, 'BODY_WRIST', ['x', 'y', 'z'])].values
                hand_wrist = output_df.loc[i, (side, 'WRIST', ['x', 'y', 'z'])].values
                thumb_tip = output_df.loc[i, (side, 'THUMB_TIP', ['x', 'y', 'z'])].values
                thumb_ip = output_df.loc[i, (side, 'THUMB_IP', ['x', 'y', 'z'])].values
                thumb_mcp = output_df.loc[i, (side, 'THUMB_MCP', ['x', 'y', 'z'])].values
                thumb_cmc = output_df.loc[i, (side, 'THUMB_CMC', ['x', 'y', 'z'])].values
                index = output_df.loc[i, (side, 'INDEX_FINGER_MCP', ['x', 'y', 'z'])].values
                ring = output_df.loc[i, (side, 'RING_FINGER_MCP', ['x', 'y', 'z'])].values
                middle = output_df.loc[i, (side, 'MIDDLE_FINGER_MCP', ['x', 'y', 'z'])].values
                pinky = output_df.loc[i, (side, 'PINKY_MCP', ['x', 'y', 'z'])].values
                elbow = output_df.loc[i, (side, 'ELBOW', ['x', 'y', 'z'])].values
                shoulder = output_df.loc[i, (side, 'SHOULDER', ['x', 'y', 'z'])].values
                lower_arm = body_wrist - elbow
                upper_arm = shoulder - elbow


                ### Wrist ##########
                # Calculate the normal vector to the plane formed by the palm
                vec1 = index - hand_wrist
                vec2 = pinky - hand_wrist
                palmNormal = np.cross(vec1, vec2)   # comes out of palm for the right arm
                palmNormal = palmNormal / np.linalg.norm(palmNormal)

                elbowNormal = np.cross(upper_arm, lower_arm)  # goes inward for the right arm

                # project the palmNormal onto the plane where the normal vector is lower_arm:
                palmNormal_proj = palmNormal - (np.dot(palmNormal, lower_arm) * lower_arm) / np.linalg.norm(lower_arm)**2

                # wrist rotation is zero when palm points inwards
                wrist_rot_abs_val = self.angleBetweenVectors(palmNormal_proj, elbowNormal)

                # right hand: angle increases when hand turns counterclockwise when viewed from outside
                # left hand: angle increases when hand turns clockwise when viewed from outside
                wrist_rot_dir = 1 if np.dot(np.cross(lower_arm, elbowNormal), palmNormal_proj) > 0 else -1
                if side == 'Left':
                    wrist_rot_dir *= -1
                angles_df.loc[i, (side, 'wristRot')] = wrist_rot_dir * wrist_rot_abs_val

                # wrist flexion is zero when the palm is parallel to the lower arm
                # it increases when the wrist is extended and decreases when it is flexed
                angles_df.loc[i, (side, 'wristFlex')] = self.angleBetweenVectors(palmNormal, -lower_arm) - 90

                ### Elbow ##########
                # 0 when fully flexed, 180 when fully extended
                angles_df.loc[i, (side, 'elbowAngle')] = self.angleBetweenVectors(lower_arm, upper_arm)

                ### Thumb ###########
                try:
                    scaler = pinky_to_index_sim / math.sqrt(np.dot((index - pinky), (index - pinky)))

                    y_axis = palmNormal
                    if side == 'Left':
                        y_axis = -y_axis
                    y = np.dot(thumb_tip, y_axis)
                    y = y - np.dot(hand_wrist, y_axis)

                    x_axis = index - pinky
                    x_axis /= np.linalg.norm(x_axis)
                    x = np.dot(thumb_tip, x_axis)
                    x = x - np.dot((ring + middle)/2, x_axis)

                    z_axis = (ring + middle)/2 - hand_wrist
                    z_axis /= np.linalg.norm(z_axis)
                    z = np.dot(thumb_tip, z_axis)
                    z = z - np.dot(hand_wrist, z_axis)

                    x *= scaler
                    y *= scaler
                    z *= scaler

                    # y*=1.5

                    # print(x, y, z)
                    targetPos = (x, y, z)
                    angles_df.loc[i, (side, 'thumb_x')] = x
                    angles_df.loc[i, (side, 'thumb_y')] = y
                    angles_df.loc[i, (side, 'thumb_z')] = z


                    # jointAngles = p.calculateInverseKinematics(handId, 15, targetPos,
                    #                                            maxNumIterations=10000, residualThreshold=0.0000001)
                    # angles_df.loc[i, (side, 'thumbOutPlaneAng')] = jointAngles[8] # thumb `rotation` angle
                    # angles_df.loc[i, (side, 'thumbInPlaneAng')] = jointAngles[9] # thumb flexion angle

                    distances = np.linalg.norm(coords - np.array(targetPos), axis=1)
                    idx = np.argmin(distances)
                    # print(angles[idx][0], angles[idx][1])
                    angles_df.loc[i, (side, 'thumbInPlaneAng')] = angles[idx][0]
                    angles_df.loc[i, (side, 'thumbOutPlaneAng')] = angles[idx][1]



                except Exception as e:
                    print(i, side, f'Error: {e}')
                    angles_df.loc[i, (side, 'thumbOutPlaneAng')] = 0  # thumb `rotation` angle
                    angles_df.loc[i, (side, 'thumbInPlaneAng')] = 0  # thumb flexion angle

                ### Hand ###########
                angles_df.loc[i, (side, 'indexAng')] = self.calculateIndex(output_df, i, side)
                angles_df.loc[i, (side, 'midAng')] = self.calculateMiddle(output_df, i, side)
                angles_df.loc[i, (side, 'ringAng')] = self.calculateRing(output_df, i, side)
                angles_df.loc[i, (side, 'pinkyAng')] = self.calculatePinky(output_df, i, side)
        angles_df.fillna(0, inplace=True)
        return angles_df







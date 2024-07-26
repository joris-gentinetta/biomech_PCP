import os.path
import numpy as np
import math
import zmq
import pybullet as p
from tqdm import tqdm
import pandas as pd
pd.options.mode.copy_on_write = True
from scipy.ndimage import gaussian_filter1d
from scipy.signal import gaussian
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.preprocessing import minmax_scale

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

        vec4 = output_df.loc[i, (side, 'INDEX_FINGER_DIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'INDEX_FINGER_PIP', ['x', 'y', 'z'])].values
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
        if os.path.exists('helpers/mapping.npz'):
            mapping = np.load('helpers/mapping.npz')
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
            np.savez('helpers/mapping.npz', coords=coords, angles=angles, pinky_to_index_sim=np.array([pinky_to_index_sim]))

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
                angles_df.loc[i, (side, 'wristRot')] = math.radians(wrist_rot_dir * wrist_rot_abs_val)

                # wrist flexion is zero when the palm is parallel to the lower arm
                # it increases when the wrist is extended and decreases when it is flexed
                angles_df.loc[i, (side, 'wristFlex')] = math.radians(self.angleBetweenVectors(palmNormal, -lower_arm) - 90)

                ### Elbow ##########
                # 0 when fully flexed, 180 when fully extended
                angles_df.loc[i, (side, 'elbowAngle')] = math.radians(self.angleBetweenVectors(lower_arm, upper_arm))

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

                    targetPos = (x, y, z)
                    angles_df.loc[i, (side, 'thumb_x')] = x
                    angles_df.loc[i, (side, 'thumb_y')] = y
                    angles_df.loc[i, (side, 'thumb_z')] = z

                    distances = np.linalg.norm(coords - np.array(targetPos), axis=1)
                    idx = np.argmin(distances)
                    angles_df.loc[i, (side, 'thumbInPlaneAng')] = angles[idx][0]
                    angles_df.loc[i, (side, 'thumbOutPlaneAng')] = angles[idx][1]



                except Exception as e:
                    print(i, side, f'Error: {e}')
                    angles_df.loc[i, (side, 'thumbOutPlaneAng')] = 0  # thumb `rotation` angle
                    angles_df.loc[i, (side, 'thumbInPlaneAng')] = 0  # thumb flexion angle

                ### Hand ###########
                angles_df.loc[i, (side, 'indexAng')] = math.radians(self.calculateIndex(output_df, i, side))
                angles_df.loc[i, (side, 'midAng')] = math.radians(self.calculateMiddle(output_df, i, side))
                angles_df.loc[i, (side, 'ringAng')] = math.radians(self.calculateRing(output_df, i, side))
                angles_df.loc[i, (side, 'pinkyAng')] = math.radians(self.calculatePinky(output_df, i, side))
        angles_df.fillna(0, inplace=True)
        return angles_df

    def mirror_pose(self, joints_df, intact_hand):
        joints_df.columns = joints_df.columns.set_names(['Side', 'Joint', 'Axis'])
        joints = joints_df[intact_hand].columns.get_level_values(0)

        affected_hand = 'Right' if intact_hand == 'Left' else 'Left'
        for joint in joints:
            joints_df.loc[:, (affected_hand, joint, 'x')] = -(joints_df.loc[:, (intact_hand, joint, 'x')] - joints_df.loc[:, (intact_hand, 'SHOULDER', 'x')]) + joints_df.loc[:, (affected_hand, 'SHOULDER', 'x')]
            joints_df.loc[:, (affected_hand, joint, 'y')] = (joints_df.loc[:, (intact_hand, joint, 'y')] - joints_df.loc[:, (intact_hand, 'SHOULDER', 'y')]) + joints_df.loc[:, (affected_hand, 'SHOULDER', 'y')]
            joints_df.loc[:, (affected_hand, joint, 'z')] = (joints_df.loc[:, (intact_hand, joint, 'z')] - joints_df.loc[:, (intact_hand, 'SHOULDER', 'z')]) + joints_df.loc[:, (affected_hand, 'SHOULDER', 'z')]

        return joints_df

    def mirror_angles(self, angles_df, intact_hand):
        angles_df.columns = angles_df.columns.set_names(['Side', 'Joint'])
        joints = angles_df[intact_hand].columns.get_level_values(0)

        affected_hand = 'Right' if intact_hand == 'Left' else 'Left'
        for joint in joints:
            if joint == 'wristRot':
                angles_df.loc[:, (affected_hand, joint)] = -angles_df.loc[:, (intact_hand, joint)]
            else:
                angles_df.loc[:, (affected_hand, joint)] = angles_df.loc[:, (intact_hand, joint)]

        return angles_df



    def apply_gaussian_smoothing(self, df, sigma, radius):
        smoothed_df = df.copy()
        for column in df.columns:
            smoothed_df[column] = gaussian_filter1d(df[column], sigma=sigma, radius=radius)
        return smoothed_df


    def print_gaussian_kernel(self, sigma, radius):
        kernel = gaussian(2*radius+1, std=sigma)
        print("Gaussian Kernel:")
        print(kernel)
        # plot it:
        plt.plot(kernel)
        plt.title(f'Gaussian Kernel with sigma={sigma} and radius={radius}')
        plt.show()

class PklConverter:
    def __init__(self, dataFolder, outputFolder, jointNames, emgChannels, side='Left'):
        self.dataFolder = dataFolder
        self.outputFolder = outputFolder

        self.jointNames = jointNames
        self.side = side

        self.emgChannels = emgChannels

        # generate the DataFrame for the angles, with the columns named appropriately
        self.columns = pd.MultiIndex.from_product([['Left', 'Right'], ['indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng', 'elbowAngle', 'wristRot', 'wristFlex']])

    def convertEMG(self, emg, saveFol):
        saveName = os.path.join(saveFol, 'cropped_aligned_emg.npy')

        emg = np.stack(emg, axis=0)

        samples, channels = emg.shape
        assert channels == len(self.emgChannels), f'Expected {len(self.emgChannels)} channels, got {channels}'

        emgOut = np.zeros((samples, 16))
        for i, channel in enumerate(self.emgChannels):
            emgOut[:, channel] = emg[:, i]

        return (saveName, emgOut)

    def convertAngles(self, angles, saveFol):
        saveName = os.path.join(saveFol, 'cropped_smooth_angles.parquet')

        angles = np.stack(angles, axis=0)

        samples, joints = angles.shape
        assert joints == len(self.jointNames), f'Expected {len(self.jointNames)} joints, got {joints}'

        # need to undo the scaling and changes applied to these angles when loaded in 4_train.py
        # the changes are as follows:
        #         data.loc[:, (args.intact_hand, 'thumbInPlaneAng')] = data.loc[:, (args.intact_hand, 'thumbInPlaneAng')] + math.pi
        #         data.loc[:, (args.intact_hand, 'wristRot')] = (data.loc[:, (args.intact_hand, 'wristRot')] + math.pi) / 2
        #         data.loc[:, (args.intact_hand, 'wristFlex')] = (data.loc[:, (args.intact_hand, 'wristFlex')] + math.pi / 2)
        #
        #         data = (2 * data - math.pi) / math.pi
        #         data = np.clip(data, -1, 1)
        #
        # Therefore, we will preempt each change in the reverse order
        #
        # We note as well that the angles from the model are _not_ in the range [-1, 1]
        # so we must make that conversion first - use a simple minmax scaler!
        angles = minmax_scale(angles, feature_range=(-1, 1))

        angles = (angles + 1) * math.pi / 2

        in_angles_df = pd.DataFrame(angles, columns=pd.MultiIndex.from_product([[self.side], self.jointNames]))
        out_angles_df = pd.DataFrame(math.pi/2, index=range(samples), columns=self.columns)

        # copy over to the main DataFrame - if a joint is all zeros, we don't want to explicitly copy as that interferes with the initializiation
        for joint in self.jointNames:
            if not np.all(np.isclose(in_angles_df.loc[:, (self.side, joint)], in_angles_df.loc[:, (self.side, joint)].iloc[0])):
                out_angles_df.loc[:, (self.side, joint)] = in_angles_df.loc[:, (self.side, joint)]

        out_angles_df.loc[:, (self.side, 'wristFlex')] = out_angles_df.loc[:, (self.side, 'wristFlex')] - math.pi/2
        out_angles_df.loc[:, (self.side, 'wristRot')] = out_angles_df.loc[:, (self.side, 'wristRot')] * 2 - math.pi
        out_angles_df.loc[:, (self.side, 'thumbInPlaneAng')] = out_angles_df.loc[:, (self.side, 'thumbInPlaneAng')] - math.pi

        return (saveName, out_angles_df)

    def genFileStructure(self, fileName):
        baseSaveFol = f'{self.outputFolder}/{fileName[:-4]}/experiments/1'
        Path(baseSaveFol).mkdir(parents=True, exist_ok=True)

        return baseSaveFol

    def save(self, saveEMG, saveAngles):
        assert saveEMG[0].endswith('.npy'), 'EMG save file must be a .npy file'
        assert saveAngles[0].endswith('.parquet'), 'Angles save file must be a .parquet file'
        assert isinstance(saveEMG[1], np.ndarray), 'EMG data must be a numpy array'
        assert isinstance(saveAngles[1], pd.DataFrame), 'Angles data must be a pandas DataFrame'
        assert saveEMG[1].shape[0] == saveAngles[1].shape[0], 'EMG and angles have different lengths'

        np.save(saveEMG[0], saveEMG[1])
        saveAngles[1].to_parquet(saveAngles[0])

    def convertPkl(self, fileName):
        saveFol = self.genFileStructure(fileName)

        with open(os.path.join(self.dataFolder, fileName), 'rb') as f:
            data = pickle.load(f)

        emg = data[0][0] if len(data[0]) == 1 else data[0]
        angles = data[1]

        assert len(emg) == len(angles), 'EMG and angles have different lengths'

        saveEMG = self.convertEMG(emg, saveFol)
        saveAngles = self.convertAngles(angles, saveFol)

        self.save(saveEMG, saveAngles)

    def pipelinePkl(self):

        for file in tqdm(os.listdir(self.dataFolder)):
            if file.endswith('.pkl'):
                self.convertPkl(file)

if __name__ == '__main__':
    # A = AnglesHelper()
    # A.print_gaussian_kernel(1.5, 2)

    pklFol = '/home/haptix/UE AMI Clinical Work/Patient Data/C1/C1_0603_2024/training/pkl/'
    outputFol = '/home/haptix/UE AMI Clinical Work/Patient Data/C1/C1_0603_2024/training/recordings/'
    emgChannels = [0, 1, 2, 4, 5, 8, 10, 11]
    jointNames = ['thumbOutPlaneAng', 'indexAng', 'midAng', 'wristFlex']

    converter = PklConverter(pklFol, outputFol, jointNames, emgChannels)
    converter.pipelinePkl()

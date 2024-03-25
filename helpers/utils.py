import numpy as np
import zmq
import pandas as pd
pd.options.mode.copy_on_write = True
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import pose, hands


class MergingHelper:
    def __init__(self):
        self.hand_keys = hands.HandLandmark._member_names_
        self.body_keys = pose.PoseLandmark._member_names_

        WRISTS = [15, 16, 17, 18, 19, 20, 21, 22]
        HEAD = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        LEGS = [25, 26, 27, 28, 29, 30, 31, 32]
        # self._non_used_landmark_ids = WRISTS + HEAD + LEGS
        self._non_used_landmark_ids = []

        self._n_landmarks_body = len(pose.PoseLandmark)  # 33
        self._n_landmarks_hand = len(hands.HandLandmark)  # 21
        self._l_hand_offset = self._n_landmarks_body - len(self._non_used_landmark_ids)
        self._r_hand_offset = self._n_landmarks_body - len(self._non_used_landmark_ids) + self._n_landmarks_hand


        self._bodyMapping = self._get_body_mapping()
        self._lHandMapping = {idx: idx + self._l_hand_offset for idx in range(self._n_landmarks_hand)}
        self._rHandMapping = {idx: idx + self._r_hand_offset for idx in range(self._n_landmarks_hand)}

        l_mapping = {name: self._lHandMapping[hands.HandLandmark[name]] for name in self.hand_keys}
        r_mapping = {name: self._rHandMapping[hands.HandLandmark[name]] for name in self.hand_keys}
        b_mapping = {name: self._bodyMapping[pose.PoseLandmark[name]] for name in self.body_keys}
        l_mapping.update(
            {'SHOULDER': b_mapping['LEFT_SHOULDER'], 'ELBOW': b_mapping['LEFT_ELBOW'], 'HIP': b_mapping['LEFT_HIP'],
                'BODY_WRIST': b_mapping['LEFT_WRIST']})
        r_mapping.update(
            {'SHOULDER': b_mapping['RIGHT_SHOULDER'], 'ELBOW': b_mapping['RIGHT_ELBOW'], 'HIP': b_mapping['RIGHT_HIP'],
                'BODY_WRIST': b_mapping['RIGHT_WRIST']})
        # these are the attributes that can be accessed from outside:
        self.map = {'Left': l_mapping, 'Right': r_mapping, 'Body': b_mapping}
        self.pose_connections = self._get_pose_connections()

    def _get_body_mapping(self):
        body_mapping = {idx: idx for idx in range(self._n_landmarks_body)}
        for idx in range(self._n_landmarks_body):
            for non_used_landmark_id in self._non_used_landmark_ids:
                if idx > non_used_landmark_id:
                    body_mapping[idx] -= 1
        for non_used_landmark_id in self._non_used_landmark_ids:
            body_mapping[non_used_landmark_id] = None
        return body_mapping

    def _get_pose_connections(self):
        # connect body:
        pose_connections = set()
        for connection in pose.POSE_CONNECTIONS:
            if connection[0] not in self._non_used_landmark_ids and connection[1] not in self._non_used_landmark_ids:
                pose_connections.add((self._bodyMapping[connection[0]], self._bodyMapping[connection[1]]))

        # connect hands:
        for connection in hands.HAND_CONNECTIONS:
            pose_connections.add((self._lHandMapping[connection[0]], self._lHandMapping[connection[1]]))
            pose_connections.add((self._rHandMapping[connection[0]], self._rHandMapping[connection[1]]))

        # connect body to hands:
        pose_connections.add(
            (self._bodyMapping[pose.PoseLandmark['LEFT_ELBOW']], self._lHandMapping[hands.HandLandmark['WRIST']]))
        pose_connections.add(
            (self._bodyMapping[pose.PoseLandmark['RIGHT_ELBOW']], self._rHandMapping[hands.HandLandmark['WRIST']]))

        return pose_connections

    def _determine_LR(self, detection_result_hands, prior=None):
        # todo lowpass filter for left/right detection

        LR_index = {'Left': None, 'Right': None}
        if detection_result_hands.hand_landmarks:
            if len(detection_result_hands.hand_landmarks) == 1:
                if prior is not None:
                    if prior == 'Left':
                        LR_index['Left'] = 0
                    elif prior == 'Right':
                        LR_index['Right'] = 0
                    else:
                        raise ValueError('Prior must be either "Left" or "Right"')
                else:
                    if detection_result_hands.handedness[0][0].category_name == "Left":
                        LR_index['Left'] = 0
                    else:
                        LR_index['Right'] = 0
            elif len(detection_result_hands.hand_landmarks) > 1:
                if detection_result_hands.handedness[0][0].score > \
                        detection_result_hands.handedness[1][0].score:
                    if detection_result_hands.handedness[0][0].category_name == "Left":
                        LR_index['Left'] = 0
                        LR_index['Right'] = 1
                    else:
                        LR_index['Right'] = 0
                        LR_index['Left'] = 1
                else:
                    if detection_result_hands.handedness[1][0].category_name == "Left":
                        LR_index['Right'] = 0
                        LR_index['Left'] = 1
                    else:
                        LR_index['Left'] = 0
                        LR_index['Right'] = 1

        return LR_index



    def mergeLandmarks(self, detection_result_hands, detection_result_body, prior=None):
        # body landmarks:
        pose_landmarks = landmark_pb2.NormalizedLandmarkList()
        if len(detection_result_body.pose_landmarks) == 0:
            for id in range(self._n_landmarks_body - len(self._non_used_landmark_ids)):
                landmark = landmark_pb2.NormalizedLandmark(x=-1, y=-1, z=-1)
                pose_landmarks.landmark.extend([landmark])
        else:
            for id, landmark in enumerate(detection_result_body.pose_landmarks[0]):
                if id in self._non_used_landmark_ids:
                    continue
                pose_landmarks.landmark.extend(
                    [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)])

        LR_index = self._determine_LR(detection_result_hands, prior)
        for hand in ['Left', 'Right']:
            if LR_index[hand] is None:
                if hand == 'Left':
                    pose_landmarks.landmark.extend([pose_landmarks.landmark[int(pose.PoseLandmark['LEFT_WRIST'])]])
                else:
                    pose_landmarks.landmark.extend([pose_landmarks.landmark[int(pose.PoseLandmark['RIGHT_WRIST'])]])

                for i in range(1, self._n_landmarks_hand):
                    landmark = landmark_pb2.NormalizedLandmark(x=-1, y=-1, z=-1)
                    pose_landmarks.landmark.extend([landmark])
            else:
                for landmark in detection_result_hands.hand_landmarks[LR_index[hand]]:
                    landmark = landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    pose_landmarks.landmark.extend([landmark])

        return pose_landmarks


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

        vec4 = output_df.loc[i, (side, 'INDEX_FINGER_PIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'INDEX_FINGER_DIP', ['x', 'y', 'z'])].values
        pipAng = self.angleBetweenVectors(vec1, vec4)

        return max(pipAng, mcpAng)


    def calculateMiddle(self, output_df, i, side):
        vec1 = output_df.loc[i, (side, 'MIDDLE_FINGER_PIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'MIDDLE_FINGER_MCP', ['x', 'y', 'z'])].values
        vec2 = output_df.loc[i, (side, 'MIDDLE_FINGER_MCP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'WRIST', ['x', 'y', 'z'])].values
        mcpAng = self.angleBetweenVectors(vec1, vec2)

        vec4 = output_df.loc[i, (side, 'MIDDLE_FINGER_PIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'MIDDLE_FINGER_DIP', ['x', 'y', 'z'])].values
        pipAng = self.angleBetweenVectors(vec1, vec4)

        return max(pipAng, mcpAng)


    def calculateRing(self, output_df, i, side):
        vec1 = output_df.loc[i, (side, 'RING_FINGER_PIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'RING_FINGER_MCP', ['x', 'y', 'z'])].values
        vec2 = output_df.loc[i, (side, 'RING_FINGER_MCP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'WRIST', ['x', 'y', 'z'])].values
        mcpAng = self.angleBetweenVectors(vec1, vec2)

        vec4 = output_df.loc[i, (side, 'RING_FINGER_PIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'RING_FINGER_DIP', ['x', 'y', 'z'])].values
        pipAng = self.angleBetweenVectors(vec1, vec4)

        return max(pipAng, mcpAng)


    def calculatePinky(self, output_df, i, side):
        vec1 = output_df.loc[i, (side, 'PINKY_PIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'PINKY_MCP', ['x', 'y', 'z'])].values
        vec2 = output_df.loc[i, (side, 'PINKY_MCP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'WRIST', ['x', 'y', 'z'])].values
        mcpAng = self.angleBetweenVectors(vec1, vec2)

        vec4 = output_df.loc[i, (side, 'PINKY_PIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'PINKY_DIP', ['x', 'y', 'z'])].values
        pipAng = self.angleBetweenVectors(vec1, vec4)

        return max(pipAng, mcpAng)


    def getArmAngles(self, output_df):
        angle_names = ['indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng', 'elbowAngle', 'wristRot',
                       'wristFlex']
        hand_names = ['Left', 'Right']

        columns = pd.MultiIndex.from_product([hand_names, angle_names])
        angles_df = pd.DataFrame(index=output_df.index, columns=columns)

        for i in angles_df.index:
            for side in ['Left', 'Right']:
                body_wrist = output_df.loc[i, (side, 'BODY_WRIST', ['x', 'y', 'z'])].values
                hand_wrist = output_df.loc[i, (side, 'WRIST', ['x', 'y', 'z'])].values
                index = output_df.loc[i, (side, 'INDEX_FINGER_MCP', ['x', 'y', 'z'])].values
                pinky = output_df.loc[i, (side, 'PINKY_MCP', ['x', 'y', 'z'])].values
                elbow = output_df.loc[i, (side, 'ELBOW', ['x', 'y', 'z'])].values
                shoulder = output_df.loc[i, (side, 'SHOULDER', ['x', 'y', 'z'])].values
                lower_arm = body_wrist - elbow
                upper_arm = shoulder - elbow


                ### Wrist ##########
                # Calculate the normal vector to the plane formed by the palm
                vec1 = index - hand_wrist
                vec2 = pinky - hand_wrist
                palmNormal = np.cross(vec1, vec2)  # comes out of palm for the right arm

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
                angles_df.loc[i, (side, 'wristFlex')] = self.angleBetweenVectors(palmNormal, lower_arm) - 90

                ### Elbow ##########
                # 0 when fully flexed, 180 when fully extended
                angles_df.loc[i, (side, 'elbowAngle')] = self.angleBetweenVectors(lower_arm, upper_arm)

                ### Thumb ###########
                thumb = output_df.loc[i, (side, 'THUMB_TIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'THUMB_MCP', ['x', 'y', 'z'])].values
                pinky_to_index = index - pinky

                thumb_plane_proj = thumb - (np.dot(thumb, palmNormal) * palmNormal) / np.linalg.norm(palmNormal)**2
                # zero when thumb is parallel the line between index and pinky, increases when thumb is pulled towards the pinky in the palm plane
                angles_df.loc[i, (side, 'thumbInPlaneAng')] = self.angleBetweenVectors(thumb_plane_proj, pinky_to_index)

                outplane_normal = np.cross(pinky_to_index, palmNormal)
                thumb_out_plane_proj = thumb - (np.dot(thumb, outplane_normal) * outplane_normal) / np.linalg.norm(outplane_normal)**2
                # zero when thumb is parallel the line between index and pinky, increases when thumb is pulled towards the pinky in the plane orthogonal to the palm
                angles_df.loc[i, (side, 'thumbOutPlaneAng')] = self.angleBetweenVectors(thumb_out_plane_proj, pinky_to_index)


                ### Hand ###########
                angles_df.loc[i, (side, 'indexAng')] = self.calculateIndex(output_df, i, side)
                angles_df.loc[i, (side, 'midAng')] = self.calculateMiddle(output_df, i, side)
                angles_df.loc[i, (side, 'ringAng')] = self.calculateRing(output_df, i, side)
                angles_df.loc[i, (side, 'pinkyAng')] = self.calculatePinky(output_df, i, side)

        return angles_df





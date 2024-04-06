import numpy as np
import zmq
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
                thumb_tip = output_df.loc[i, (side, 'THUMB_TIP', ['x', 'y', 'z'])].values
                thumb_ip = output_df.loc[i, (side, 'THUMB_IP', ['x', 'y', 'z'])].values
                thumb_mcp = output_df.loc[i, (side, 'THUMB_MCP', ['x', 'y', 'z'])].values
                thumb_cmc = output_df.loc[i, (side, 'THUMB_CMC', ['x', 'y', 'z'])].values
                index = output_df.loc[i, (side, 'INDEX_FINGER_MCP', ['x', 'y', 'z'])].values
                ring = output_df.loc[i, (side, 'RING_FINGER_MCP', ['x', 'y', 'z'])].values
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
                angles_df.loc[i, (side, 'wristFlex')] = self.angleBetweenVectors(palmNormal, -lower_arm) - 90

                ### Elbow ##########
                # 0 when fully flexed, 180 when fully extended
                angles_df.loc[i, (side, 'elbowAngle')] = self.angleBetweenVectors(lower_arm, upper_arm)

                ### Thumb ###########
                # thumb = thumb_tip - thumb_cmc # defines the line of the thumb
                # thumb_index = index - thumb_mcp
                index_wrist = index - hand_wrist
                thumb_cmc_angle = thumb_cmc - hand_wrist
                thumb_distal = thumb_tip - thumb_ip
                thumb_proximal = thumb_ip - thumb_mcp
                thumb_base = thumb_mcp - thumb_cmc
                thumb_link = thumb_tip - thumb_mcp
                # pinky_to_index = index - pinky
                cmc_to_ring = ring - thumb_cmc # defines the axis of rotation for the thumb

                thumb_plane_normal = np.cross(cmc_to_ring, thumb_cmc_angle) # normal to the plane formed by the thumb and the axis of rotation
                thumb_rot_angle = -3*(self.angleBetweenVectors(thumb_plane_normal, palmNormal) - 20) # angle between the normal to the thumb plane and the normal to the palm plane (negate for psyonic)

                thumb_flex_angle = 3*(np.max(90 - np.asarray([self.angleBetweenVectors(thumb_distal, -thumb_proximal), self.angleBetweenVectors(thumb_link, index_wrist), self.angleBetweenVectors(thumb_base, index_wrist)])) - 50) # angle between the thumb and the index finger

                # zero when thumb is parallel the line between index and pinky, increases when thumb is pulled towards the pinky in the palm plane
                angles_df.loc[i, (side, 'thumbInPlaneAng')] = thumb_flex_angle # thumb flexion angle

                # zero when thumb is parallel the line between index and pinky, increases when thumb is pulled towards the pinky in the plane orthogonal to the palm
                angles_df.loc[i, (side, 'thumbOutPlaneAng')] = thumb_rot_angle # thumb `rotation` angle

                # thumb = output_df.loc[i, (side, 'THUMB_TIP', ['x', 'y', 'z'])].values - output_df.loc[i, (side, 'THUMB_MCP', ['x', 'y', 'z'])].values
                # pinky_to_index = index - pinky

                # thumb_plane_proj = thumb - (np.dot(thumb, palmNormal) * palmNormal) / np.linalg.norm(palmNormal)**2
                # # zero when thumb is parallel the line between index and pinky, increases when thumb is pulled towards the pinky in the palm plane
                # angles_df.loc[i, (side, 'thumbInPlaneAng')] = self.angleBetweenVectors(thumb_plane_proj, pinky_to_index)

                # outplane_normal = np.cross(pinky_to_index, palmNormal)
                # thumb_out_plane_proj = thumb - (np.dot(thumb, outplane_normal) * outplane_normal) / np.linalg.norm(outplane_normal)**2
                # # zero when thumb is parallel the line between index and pinky, increases when thumb is pulled towards the pinky in the plane orthogonal to the palm
                # angles_df.loc[i, (side, 'thumbOutPlaneAng')] = self.angleBetweenVectors(thumb_out_plane_proj, pinky_to_index)

                ### Hand ###########
                angles_df.loc[i, (side, 'indexAng')] = self.calculateIndex(output_df, i, side)
                angles_df.loc[i, (side, 'midAng')] = self.calculateMiddle(output_df, i, side)
                angles_df.loc[i, (side, 'ringAng')] = self.calculateRing(output_df, i, side)
                angles_df.loc[i, (side, 'pinkyAng')] = self.calculatePinky(output_df, i, side)

        return angles_df





import argparse
from os.path import join

import numpy as np
import pandas as pd
pd.options.mode.copy_on_write = True
idx = pd.IndexSlice
import os
from tqdm import tqdm
import ast
from helpers.utils import restore_multiindex_columns
import shutil

class ZeroJoints:
    def __init__(self, intact_hand):
        self.intact_hand = intact_hand

        self.anglesMapping = {
            'index': ['indexAng'],
            'mrp': ['midAng', 'ringAng', 'pinkyAng'],
            'thumb': ['thumbInPlaneAng', 'thumbOutPlaneAng'],
            'wrist': ['wristFlex'],
            'thumbFlEx': ['thumbInPlaneAng', 'thumbOutPlaneAng'],
            'thumbAbAd': ['thumbInPlaneAng', 'thumbOutPlaneAng'],
            'thumbOpp': ['thumbInPlaneAng', 'thumbOutPlaneAng'],
            'indexFlEx': ['indexAng'],
            'mrpFlEx': ['midAng', 'ringAng', 'pinkyAng'],
            'fingersFlEx': ['indexAng', 'midAng', 'ringAng', 'pinkyAng'],
            'handOpCl': ['indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng'],
            'pinchOpCl': ['indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng'],
            'pointOpCl': ['indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng'],
            'keyOpCl': ['indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng'],
            'indexFlDigitsEx': ['indexAng', 'midAng', 'ringAng', 'pinkyAng'],
            'wristFlEx': ['wristFlex'],
            'wristFlHandCl': ['wristFlex', 'indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng'],
        }

    def zero_out_joints(self, angles_df, angles_used):
        colIdx = [(self.intact_hand, ang) for ang in angles_used]
        keep_df = angles_df.loc[:, colIdx].copy()
        angles_df = pd.DataFrame(0, index=angles_df.index, columns=angles_df.columns)

        angles_df = (angles_df * np.pi + np.pi) / 2
        angles_df.loc[:, (self.intact_hand, 'wristFlex')] = angles_df.loc[:, (self.intact_hand, 'wristFlex')] - np.pi / 2
        angles_df.loc[:, (self.intact_hand, 'wristRot')] = (angles_df.loc[:, (self.intact_hand, 'wristRot')] * 2) - np.pi
        angles_df.loc[:, (self.intact_hand, 'thumbInPlaneAng')] = angles_df.loc[:,
                                                             (self.intact_hand, 'thumbInPlaneAng')] - np.pi

        angles_df.loc[:, colIdx] = keep_df
        angles_df.columns = angles_df.columns.set_names(['Side', 'Joint'])

        return angles_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zero out the joints as appropriate to improve predictions.')
    parser.add_argument('--data_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--intact_hand', type=str, required=True, help='Intact hand (Right/Left)')
    args = parser.parse_args()

    if not args.intact_hand in ['Right', 'Left']:
        raise ValueError("Intact hand must be 'Right' or 'Left'")
    intact_hand = args.intact_hand

    zeroJoints = ZeroJoints(intact_hand)

    recordings_dir = join(args.data_dir, 'recordings (copy)')
    for exp in tqdm(os.listdir(recordings_dir)):
        angles_file = join(recordings_dir, exp, 'experiments/1/cropped_smooth_angles.parquet')

        # save as an _orig file
        # orig_file = angles_file.replace('.parquet', '_orig.parquet')
        shutil.copy(angles_file, angles_file.replace('.parquet', '_orig.parquet'))

        if not os.path.exists(angles_file):
            continue

        orig_df = pd.read_parquet(angles_file)
        orig_df.columns = orig_df.columns.set_names(['Side', 'Joint'])

        if isinstance(orig_df.columns[0], str) and orig_df.columns[0].startswith("("):
            try:
                orig_df.columns = pd.MultiIndex.from_tuples([ast.literal_eval(col) for col in orig_df.columns])
            except Exception:
                raise ValueError("Columns look like stringified tuples but couldn't be parsed. Check format.")

        # colIdx = idx[intact_hand, angles_used]
        # print(orig_df.columns)
        # angles_used = zeroJoints.anglesMapping[exp]
        # colIdx = [(intact_hand, ang) for ang in angles_used]
        #
        # # then the ones that arent angles_used (on the intact hand) should be zeroed out
        # keep_df = orig_df.loc[:, colIdx].copy()
        # angles_df = pd.DataFrame(0, index=orig_df.index, columns=orig_df.columns)
        #
        # angles_df = (angles_df * np.pi + np.pi) / 2
        # angles_df.loc[:, (intact_hand, 'wristFlex')] = angles_df.loc[:, (intact_hand, 'wristFlex')] - np.pi / 2
        # angles_df.loc[:, (intact_hand, 'wristRot')] = (angles_df.loc[:, (intact_hand, 'wristRot')] * 2) - np.pi
        # angles_df.loc[:, (intact_hand, 'thumbInPlaneAng')] = angles_df.loc[:,
        #                                                      (intact_hand, 'thumbInPlaneAng')] - np.pi
        #
        # angles_df.loc[:, colIdx] = keep_df
        # angles_df.columns = angles_df.columns.set_names(['Side', 'Joint'])

        angles_df = zeroJoints.zero_out_joints(orig_df, zeroJoints.anglesMapping[exp])
        angles_df.to_parquet(angles_file, index=False)
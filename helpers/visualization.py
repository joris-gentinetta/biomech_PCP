# --------------------------------------------------------------------------------
# BodyFlow
# Version: 1.0
# Copyright (c) 2023 Instituto Tecnologico de Aragon (www.itainnova.es) (Spain)
# Date: February 2023
# Authors: Ana Caren Hernandez Ruiz                      ahernandez@itainnova.es
#          Angel Gimeno Valero                              agimeno@itainnova.es
#          Carlos Maranes Nueno                            cmaranes@itainnova.es
#          Irene Lopez Bosque                                ilopez@itainnova.es
#          Maria de la Vega Rodrigalvarez Chamarro   vrodrigalvarez@itainnova.es
#          Pilar Salvo Ibanez                                psalvo@itainnova.es
#          Rafael del Hoyo Alonso                          rdelhoyo@itainnova.es
#          Rocio Aznar Gimeno                                raznar@itainnova.es
# All rights reserved 
# --------------------------------------------------------------------------------


import logging
import cv2
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.animation as animation
if sys.platform != 'darwin':
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    rcParams['animation.ffmpeg_path'] = r'/home/haptix/anaconda3/bin/ffmpeg'
import matplotlib.pyplot as plt
from matplotlib import gridspec
from os.path import join
import pandas as pd
idx = pd.IndexSlice

logging.getLogger('matplotlib').setLevel(logging.WARNING)





class Visualization():
    def __init__(self, data_dir, df_3d, alternative=True, name_addition=""):
        """
        This function takes as an input the video and outputs a video 
        with the original video with the 2D joints and the 3D plot.  
        """
        self.data_dir = data_dir
        self.name_addition = name_addition
        self.video_fnm = join(data_dir, 'cropped_video.mp4')
        self.df_3d = df_3d.copy().sort_index(axis=1)
        self.df_3d.loc[:, idx[slice(None), slice(None), 'z']] = self.df_3d.loc[:, idx[slice(None), slice(None), 'z']] - np.min(self.df_3d.loc[:, idx[slice(None), slice(None), 'z']]) - 500
        # self.df_3d.loc[:, idx[slice(None), slice(None), 'x']] = self.df_3d.loc[:, idx[slice(None), slice(None), 'x']] - np.min(self.df_3d.loc[:, idx[slice(None), slice(None), 'x']])
        # self.df_3d.loc[:, idx[slice(None), slice(None), 'y']] = self.df_3d.loc[:, idx[slice(None), slice(None), 'y']] - np.min(self.df_3d.loc[:, idx[slice(None), slice(None), 'y']])
        self.number_frames = len(self.df_3d.index)
        self.alternative = alternative
        # video_frames, fps = self.get_video_frames()
        # self.video_frames = video_frames
        # self.fps = fps
        self.vcap, self.fps = self.get_video_frames()
        self.plotVideo()



    def get_video_frames(self):
        """
        Fetchs frames of the video input with opencv and the frames per second
        """
        vcap = cv2.VideoCapture(self.video_fnm)
        fps = vcap.get(cv2.CAP_PROP_FPS)
        logging.info('Video frames processed')
        return vcap, fps

    def get_arm(self, side, i):
        x = self.df_3d.loc[i, (side, ('SHOULDER', 'ELBOW', 'BODY_WRIST'), 'x')].values
        y = self.df_3d.loc[i, (side, ('SHOULDER', 'ELBOW', 'BODY_WRIST'), 'y')].values
        z = self.df_3d.loc[i, (side, ('SHOULDER', 'ELBOW', 'BODY_WRIST'), 'z')].values

        return x, y, z

    def get_body(self, i):
        x = np.append(self.df_3d.loc[i, ('Body', ('RIGHT_HIP', 'LEFT_HIP', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'RIGHT_HIP'), 'x')].values, self.df_3d.loc[i, ('Body', 'RIGHT_HIP', 'x')])
        y = np.append(self.df_3d.loc[i, ('Body', ('RIGHT_HIP', 'LEFT_HIP', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'RIGHT_HIP'), 'y')].values, self.df_3d.loc[i, ('Body', 'RIGHT_HIP', 'y')])
        z = np.append(self.df_3d.loc[i, ('Body', ('RIGHT_HIP', 'LEFT_HIP', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'RIGHT_HIP'), 'z')].values, self.df_3d.loc[i, ('Body', 'RIGHT_HIP', 'z')])

        return x, y, z

    def get_hand(self, side, i):
        hand= [
            'WRIST',
            'THUMB_CMC',
            'THUMB_MCP',
            'THUMB_IP',
            'THUMB_TIP',
            'THUMB_IP',
            'THUMB_MCP',
            'THUMB_CMC',

            'WRIST',
            'INDEX_FINGER_MCP',
            'INDEX_FINGER_PIP',
            'INDEX_FINGER_DIP',
            'INDEX_FINGER_TIP',
            'INDEX_FINGER_DIP',
            'INDEX_FINGER_PIP',
            'INDEX_FINGER_MCP',

            'MIDDLE_FINGER_MCP',
            'MIDDLE_FINGER_PIP',
            'MIDDLE_FINGER_DIP',
            'MIDDLE_FINGER_TIP',
            'MIDDLE_FINGER_DIP',
            'MIDDLE_FINGER_PIP',
            'MIDDLE_FINGER_MCP',

            'RING_FINGER_MCP',
            'RING_FINGER_PIP',
            'RING_FINGER_DIP',
            'RING_FINGER_TIP',
            'RING_FINGER_DIP',
            'RING_FINGER_PIP',
            'RING_FINGER_MCP',

            'PINKY_MCP',
            'PINKY_PIP',
            'PINKY_DIP',
            'PINKY_TIP',
            'PINKY_DIP',
            'PINKY_PIP',
            'PINKY_MCP',
            'WRIST',
        ]

        x = [self.df_3d.loc[i, (side, joint, 'x')] for joint in hand]
        y = [self.df_3d.loc[i, (side, joint, 'y')] for joint in hand]
        z = [self.df_3d.loc[i, (side, joint, 'z')] for joint in hand]

        return np.array(x), np.array(y), np.array(z)


    def update_lines(self, i, pbar):
        pbar.update(1)
        x_h_l, y_h_l, z_h_l = self.get_hand('Left', i)
        x_l, y_l, z_l = self.get_arm('Left', i)

        x_h_r, y_h_r, z_h_r = self.get_hand('Right', i)
        x_r, y_r, z_r = self.get_arm('Right', i)

        x_b, y_b, z_b = self.get_body(i)

        # Update 3D data of the current frame
        self.clj1[0].set_data_3d(x_l, z_l, -y_l)
        self.crj1[0].set_data_3d(x_r, z_r, -y_r)
        self.cbj1[0].set_data_3d(x_b, z_b, -y_b)

        # Update 3D data of the current frame second view
        self.clj2[0].set_data_3d(x_l, z_l, -y_l)
        self.crj2[0].set_data_3d(x_r, z_r, -y_r)
        self.cbj2[0].set_data_3d(x_b, z_b, -y_b)

        x_l = np.concatenate((x_l, x_h_l))
        y_l = np.concatenate((y_l, y_h_l))

        x_r = np.concatenate((x_r, x_h_r))
        y_r = np.concatenate((y_r, y_h_r))

        self.clj12d[0].set_data(x_l, y_l)
        self.crj12d[0].set_data(x_r, y_r)
        self.cbj12d[0].set_data(x_b, y_b)

        # Update the video frame
        success, video_frame = self.vcap.read()
        if success:
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            self.vplot.set_data(video_frame)
        else:
            print(f"Can't read video frame {i}")

        return 


    def initVideo(self):
        i = 0
        fig = plt.figure(figsize=(22,12))
        spec = gridspec.GridSpec(ncols = 3, nrows = 1, width_ratios = [1.8, 1, 1],
                                wspace = 0.1, hspace = 0.1 )

        # Video frames plot
        axv = fig.add_subplot(spec[0])
        video_frame = self.vcap.read()[1]
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        self.vplot = axv.imshow(video_frame)
        axv.set_xlim(0, video_frame.shape[1])
        axv.set_ylim(video_frame.shape[0], 0)
        axv.set_title(f'Original Video + 2D', fontweight='bold')
        axv.axis('off')
        
        # Linewidth for joints
        lw = 1

        x_h_l, y_h_l, z_h_l = self.get_hand('Left', i)
        x_l, y_l, z_l = self.get_arm('Left', i)

        x_h_r, y_h_r, z_h_r = self.get_hand('Right', i)
        x_r, y_r, z_r = self.get_arm('Right', i)

        x_b, y_b, z_b = self.get_body(i)

        # 3D plots 
        self.ax1=fig.add_subplot(spec[1], projection='3d')
        self.ax1.set_title(f'3D Front View', fontweight='bold')
        self.ax1.set_box_aspect([1,1,1])
        self.ax1.view_init(elev = 20, azim = -90) # For better visualization
        self.ax1.set_xticklabels([])
        self.ax1.set_yticklabels([])
        self.ax1.set_zticklabels([])


        # 3D plots 2nd view
        self.ax2 = fig.add_subplot(spec[2], projection='3d')
        self.ax2.set_title(f'3D Side View', fontweight='bold')
        self.ax2.set_box_aspect([1, 1, 1])
        self.ax2.view_init(elev=20, azim=-180)
        self.ax2.set_xticklabels([])
        self.ax2.set_yticklabels([])
        self.ax2.set_zticklabels([])

        x_lim= [0, video_frame.shape[1]]
        y_lim= [-video_frame.shape[0], 0]
        z_lim = x_lim

        self.ax1.set_xlim3d(x_lim)
        self.ax1.set_ylim3d(z_lim)
        self.ax1.set_zlim3d(y_lim)

        self.ax2.set_xlim3d(x_lim)
        self.ax2.set_ylim3d(z_lim)
        self.ax2.set_zlim3d(y_lim)

        # Plot 3D joints
        self.clj1 = self.ax1.plot(x_l, z_l, -y_l, c=(1, 0, 0), linewidth=lw, zdir='z')
        self.crj1 = self.ax1.plot(x_r, z_r, -y_r, c=(0, 1, 0), linewidth=lw, zdir='z')
        self.cbj1 = self.ax1.plot(x_b, z_b, -y_b, c=(0, 0, 0), linewidth=lw, zdir='z')

        # Plot 3D joints
        self.clj2 = self.ax2.plot(x_l, z_l, -y_l, c=(1, 0, 0), linewidth=lw, zdir='z')
        self.crj2 = self.ax2.plot(x_r, z_r, -y_r, c=(0, 1, 0), linewidth=lw, zdir='z')
        self.cbj2 = self.ax2.plot(x_b, z_b, -y_b, c=(0, 0, 0), linewidth=lw, zdir='z')

        x_l = np.concatenate((x_l, x_h_l))
        y_l = np.concatenate((y_l, y_h_l))

        x_r = np.concatenate((x_r, x_h_r))
        y_r = np.concatenate((y_r, y_h_r))

        # Fetch and plot 2D data
        self.clj12d = axv.plot(x_l, y_l, c = (1,0,0), linewidth = lw)
        self.crj12d = axv.plot(x_r, y_r, c = (0,1,0), linewidth = lw)
        self.cbj12d = axv.plot(x_b, y_b, c = (0,0,0), linewidth = lw)

        logging.info("Plotting...")
        return fig 


    def plotVideo(self):

        # Animation
        fig = self.initVideo()
        print('Creating animation...')
        pbar = tqdm(total=self.number_frames)

        anim = animation.FuncAnimation(fig, lambda i: self.update_lines(i, pbar),
                                            frames = range(1, self.number_frames), interval = 1,
                                            blit = False, repeat = False, cache_frame_data = False)
        writervideo = animation.FFMpegWriter(fps=self.fps, codec="libx264")
        anim.save(join(self.data_dir, f'visualization{self.name_addition}.mp4'), writer = writervideo)


        logging.info('Video saved!')
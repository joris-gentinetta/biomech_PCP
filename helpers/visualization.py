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

#  Code for visualization of a single output

import os
import logging
import sys
import numpy as np
import matplotlib.animation as animation
if sys.platform != 'darwin':
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    rcParams['animation.ffmpeg_path'] = r'/home/haptix/anaconda3/bin/ffmpeg'
import matplotlib.pyplot as plt
from matplotlib import gridspec
from os.path import join
logging.getLogger('matplotlib').setLevel(logging.WARNING)


class Visualization():
    def __init__(self, data_dir, df_2d, df_3d, alternative=True, name_addition=""):
        """
        This function takes as an input the video and outputs a video 
        with the original video with the 2D joints and the 3D plot.  
        """
        self.data_dir = data_dir
        self.name_addition = name_addition
        self.video_fnm = join(data_dir, 'cropped_video.mp4')
        self.df_2d = df_2d
        self.df_3d = df_3d
        self.number_frames = len(self.df_2d.index)
        self.alternative = alternative
        video_frames, fps = self.get_video_frames()
        self.video_frames = video_frames
        self.fps = fps
        self.plotVideo()
        

    def get_video_frames(self):
        """
        Fetchs frames of the video input with opencv and the frames per second
        """       
        import cv2
        vcap = cv2.VideoCapture(self.video_fnm)
        fps = vcap.get(cv2.CAP_PROP_FPS)
        frames = []
        ret = True
        while ret:
            ret, img = vcap.read() 
            if ret:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                frames.append(img)
        video_frames = np.stack(frames, axis=0)
        # video_frames = video_frames[self.df.iloc[:,0][0]-1: self.df.iloc[:,0][len(self.df)-1]]
        logging.info('Video frames processed')
        return video_frames, fps
    
    def get3djoints(self):
        """
        Gets the 3d keyoints from the CSV by reading the headers.
        """
        joint_names = ["HEAD", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
         "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
         "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "JAW", "CHEST", "SPINE", "HIPS" ]
        n_frames = len(self.df_3d.index)

        axes = ['x', 'y', 'z']
        joints = np.zeros((len(self.df_3d), len(joint_names), len(axes)))
        for frame in range(n_frames):
            for i, joint in enumerate(joint_names):
                for j, ax in enumerate(axes):
                    # joints[:, i, j] = np.asarray(self.df.iloc[:, self.df.columns.get_loc(f'{joint}.coordinate_{ax}')])
                    joints[frame, i, j] = np.asarray(self.df_3d.loc[frame, ('Body', joint, ax)])

        return joints

    def get2djoints(self):
        """
        Gets the 2d keyoints froms the CSV by reading the headers. 
        """   
        joint_names = ['HIPS', 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE', 'SPINE',
                       'CHEST', 'JAW', 'HEAD', 'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
        hands = [
    'WRIST',
    'THUMB_CMC',
    'THUMB_MCP',
    'THUMB_IP',
    'THUMB_TIP',
    'INDEX_FINGER_MCP',
    'INDEX_FINGER_PIP',
    'INDEX_FINGER_DIP',
    'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP',
    'MIDDLE_FINGER_PIP',
    'MIDDLE_FINGER_DIP',
    'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP',
    'RING_FINGER_PIP',
    'RING_FINGER_DIP',
    'RING_FINGER_TIP',
    'PINKY_MCP',
    'PINKY_PIP',
    'PINKY_DIP',
    'PINKY_TIP'
]

        n_frames = len(self.df_2d.index)

        axes = ['x', 'y']
        joints = np.zeros((len(self.df_2d), len(joint_names) + 2 * len(hands), len(axes)))
        for frame in range(n_frames):
            for i, joint in enumerate(joint_names):
                for j, ax in enumerate(axes):
                    joints[frame, i, j] = np.asarray(self.df_2d.loc[frame, ('Body', joint, ax)])
            for i, joint in enumerate(hands):
                for j, ax in enumerate(axes):
                    joints[frame, i + len(joint_names), j] = np.asarray(self.df_2d.loc[frame, ('Left', joint, ax)])
                    joints[frame, i + len(joint_names) + len(hands), j] = np.asarray(self.df_2d.loc[frame, ('Right', joint, ax)])

        return joints

    def get_arm(self, side, i):
        x = self.df_3d.loc[i, (side, ('SHOULDER', 'ELBOW', 'WRIST'), 'x')].values
        y = self.df_3d.loc[i, (side, ('SHOULDER', 'ELBOW', 'WRIST'), 'y')].values
        z = self.df_3d.loc[i, (side, ('SHOULDER', 'ELBOW', 'WRIST'), 'z')].values

        return x, y, z

    def get_body(self, i):
        x = np.append(self.df_3d.loc[i, ('Body', ('RIGHT_HIP', 'LEFT_HIP', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'RIGHT_HIP'), 'x')].values, self.df_3d.loc[i, ('Body', 'RIGHT_HIP', 'x')])
        y = np.append(self.df_3d.loc[i, ('Body', ('RIGHT_HIP', 'LEFT_HIP', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'RIGHT_HIP'), 'y')].values, self.df_3d.loc[i, ('Body', 'RIGHT_HIP', 'y')])
        z = np.append(self.df_3d.loc[i, ('Body', ('RIGHT_HIP', 'LEFT_HIP', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'RIGHT_HIP'), 'z')].values, self.df_3d.loc[i, ('Body', 'RIGHT_HIP', 'z')])

        return x, y, z

    def update_lines(self, i):
        x_r, y_r, z_r = self.get_arm('Right', i)
        x_l, y_l, z_l = self.get_arm('Left', i)
        x_b, y_b, z_b = self.get_body(i)


        self.clj12d[0].set_data(x_l, y_l)
        self.crj12d[0].set_data(x_r, y_r)
        self.cbj12d[0].set_data(x_b, y_b)

        
        # Fetch 3D data of the current frame 
        # xs1, ys1, zs1 = self.x1_3d[i][:, 0], self.x1_3d[i][:, 2], -self.x1_3d[i][:, 1]

        # Update 3D data of the current frame
        self.clj1[0].set_data_3d(x_l, z_l, -y_l)
        self.crj1[0].set_data_3d(x_r, z_r, -y_r)
        self.cbj1[0].set_data_3d(x_b, z_b, -y_b)

        # Update 3D data of the current frame second view
        self.clj2[0].set_data_3d(x_l, z_l, -y_l)
        self.crj2[0].set_data_3d(x_r, z_r, -y_r)
        self.cbj2[0].set_data_3d(x_b, z_b, -y_b)

        """
        Updates the limits if alternative is True, this parameter makes your grid adapt
        to the current maximum and minimum of the joints.
        """
        if self.alternative == True:
            coords = np.concatenate((x_l, x_r, x_b, -y_l, -y_r, -y_b, z_l, z_r, z_b))
            min_coord = np.amin(coords)
            max_coord = np.amax(coords)

            self.ax1.set_xlim3d([min_coord, max_coord])
            self.ax1.set_ylim3d([min_coord, max_coord])
            self.ax1.set_zlim3d([min_coord, max_coord])

            self.ax2.set_xlim3d([min_coord, max_coord])
            self.ax2.set_ylim3d([min_coord, max_coord])
            self.ax2.set_zlim3d([min_coord, max_coord])


        # Update the video frame
        self.vplot.set_data(self.video_frames[i])

        return 


    def initVideo(self):
        i = 0
        fig = plt.figure(figsize=(11,6))
        spec = gridspec.GridSpec(ncols = 3, nrows = 1, width_ratios = [1.8, 1, 1],
                                wspace = 0.1, hspace = 0.1 )

        # Video frames plot
        axv = fig.add_subplot(spec[0])
        self.vplot = axv.imshow(self.video_frames[i])
        axv.set_xlim(0, self.video_frames[i].shape[1])
        axv.set_ylim(self.video_frames[i].shape[0], 0)
        axv.set_title(f'Original Video + 2D', fontweight = 'bold' )
        axv.axis('off')
        
        # Linewidth for joints
        lw = 1

        x_l, y_l, z_l = self.get_arm('Left', i)
        x_r, y_r, z_r = self.get_arm('Right', i)
        x_b, y_b, z_b = self.get_body(i)

        # Fetch and plot 2D data
        self.clj12d = axv.plot(x_l, y_l, c = (1,0,0), linewidth = lw)
        self.crj12d = axv.plot(x_r, y_l, c = (0,1,0), linewidth = lw)
        self.cbj12d = axv.plot(x_b, y_b, c = (0,0,0), linewidth = lw)

        
        # 3D plots 
        self.ax1=fig.add_subplot(spec[1], projection='3d')
        self.ax1.set_title(f'3D Front View', fontweight='bold')
        self.ax1.set_box_aspect([1,1,1])
        self.ax1.view_init(elev = 0, azim = -90) # For better visualization
        self.ax1.set_xticklabels([])
        self.ax1.set_yticklabels([])
        self.ax1.set_zticklabels([])


        # 3D plots 2nd view
        self.ax2 = fig.add_subplot(spec[2], projection='3d')
        self.ax2.set_title(f'3D Side View', fontweight='bold')
        self.ax2.set_box_aspect([1, 1, 1])
        self.ax2.view_init(elev=0, azim=-180)
        self.ax2.set_xticklabels([])
        self.ax2.set_yticklabels([])
        self.ax2.set_zticklabels([])


        if self.alternative == True:
            # The current min/max of the skeleton joints
            coords = np.concatenate((x_l, x_r, x_b, -y_l, -y_r, -y_b, z_l, z_r, z_b))
            min_coord = np.amin(coords)
            max_coord = np.amax(coords)

        else:
            # The global min/max of the whole sequence
            max_coord = self.df_3d.loc[:, (slice(None), slice(None), slice(None))].max().max()
            min_coord = self.df_3d.loc[:, (slice(None), slice(None), slice(None))].min().min()

        self.ax2.set_xlim3d([min_coord, max_coord])
        self.ax2.set_ylim3d([min_coord, max_coord])
        self.ax2.set_zlim3d([min_coord, max_coord])

        self.ax1.set_xlim3d([min_coord, max_coord])
        self.ax1.set_ylim3d([min_coord, max_coord])
        self.ax1.set_zlim3d([min_coord, max_coord])

        # Plot 3D joints
        self.clj1 = self.ax1.plot(x_l, z_l, -y_l, c=(1, 0, 0), linewidth=lw, zdir='z')
        self.crj1 = self.ax1.plot(x_r, z_r, -y_r, c=(0, 1, 0), linewidth=lw, zdir='z')
        self.cbj1 = self.ax1.plot(x_b, z_b, -y_b, c=(0,0,0), linewidth=lw, zdir='z')

        # Plot 3D joints
        self.clj2 = self.ax2.plot(x_l, z_l, -y_l, c=(1, 0, 0), linewidth=lw, zdir='z')
        self.crj2 = self.ax2.plot(x_r, z_r, -y_r, c=(0, 1, 0), linewidth=lw, zdir='z')
        self.cbj2 = self.ax2.plot(x_b, z_b, -y_b, c=(0,0,0), linewidth=lw, zdir='z')

        logging.info("Plotting...")
        return fig 


    def plotVideo(self):
        # Joints order for plotting for 2D and 3D sequences
        self.l_joints2d = [13, 12, 11, 8, 7, 0, 4, 5, 6]
        self.r_joints2d = [16, 15, 14, 8, 9, 10, 9, 8, 14, 15, 16, 15, 14, 8, 7, 0, 1, 2, 3]
        self.l_hand = [i + 17 for i in [0, 1, 2, 3, 4, 3, 2, 1, 0, 5, 6, 7, 8, 7, 6, 5, 9, 10, 11, 12, 11, 10, 9, 13, 14, 15, 16, 15, 14, 13, 17, 18, 19, 20, 19, 18, 17]]
        self.r_hand = [i + 21 for i in self.l_hand]
        self.l_joints2d = self.l_hand + self.l_joints2d
        self.r_joints2d = self.r_hand + self.r_joints2d
                
        self.l_joints = [5, 3, 1, 14, 15, 16, 7, 9, 11]
        self.r_joints = [0, 13, 14, 2, 4, 6, 4, 2, 14, 15, 16, 8, 10, 12]

        # Animation
        fig = self.initVideo()
        anim = animation.FuncAnimation(fig, self.update_lines,
                                        frames = self.number_frames, interval = 1, 
                                        blit = False, repeat = False, cache_frame_data = False)
        writervideo = animation.FFMpegWriter(fps=self.fps, codec="libx264")
        anim.save(join(self.data_dir, f'visualization{self.name_addition}.mp4'), writer = writervideo)
        logging.info('Video saved!')
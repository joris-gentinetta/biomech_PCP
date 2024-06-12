# BUGMAN Feb 1 2022
# Modified by Mikey Fernandez 06/09/2022

"""
The parent class for a 4 DoF upper extremity model - accepts multiple types of model
"""

from re import I
import torch
from torch import nn
from torch.functional import F
import numpy as np

class Hand_4dof(nn.Module):
    def __init__(self, muscleType, device, EMG_Channel_Count, Dynamic_Lr, EMG_mat_Lr, NN_ratio):
        super().__init__()
        
        self.device = device
        self.EMG_Channel_Count = EMG_Channel_Count
        self.EMG_mat_Lr = EMG_mat_Lr

        # make dictionary of joint model inits
        jointModels = {'bilinear': self.bilinearInit, 'shadmehr': self.shadmehrInit, 'thelen': self.thelenInit}

        # import the correct joint model
        if muscleType == "bilinear":
            from Dynamics2.Joint_1dof_Bilinear_NN import Joint_1dof
        elif muscleType == "shadmehr":
            from Dynamics2.Joint_1dof_Shadmehr import Joint_1dof
        elif muscleType == "thelen":
            from Dynamics2.Joint_1dof_Thelen import Joint_1dof
        else:
            raise ValueError(f'Invalid muscleType {muscleType}')

        jointModels[muscleType]()
        
        self.muscles = [self.m1, self.m2, self.m3, self.m4, self.m5, self.m6, self.m7, self.m8]
         
        self.AMI1 = self.muscles[0:2]
        self.AMI2 = self.muscles[2:4]
        self.AMI3 = self.muscles[4:6]
        self.AMI4 = self.muscles[6:8]
        
        self.muscle_num = len(self.muscles)

        # Joints
        self.Joint1 = Joint_1dof(self.device, self.AMI1, self.params, Dynamic_Lr, NN_ratio)
        self.Joint2 = Joint_1dof(self.device, self.AMI2, self.params, Dynamic_Lr, NN_ratio)
        self.Joint3 = Joint_1dof(self.device, self.AMI3, self.params, Dynamic_Lr, NN_ratio)
        self.Joint4 = Joint_1dof(self.device, self.AMI4, self.params, Dynamic_Lr, NN_ratio)

        # EMG to Muscle activation matrix
        # If the electrodes are right upon the targeted muscle, eye matrix should be chose, otherwise all-one matrix should be the way to go
        # I'm using the scaled all-one matrix implementation here.
        self.EMG_to_Activation_Mat = nn.Parameter(torch.ones((self.EMG_Channel_Count, self.muscle_num), dtype=torch.float, device=self.device)/self.EMG_mat_Lr/self.EMG_Channel_Count)
        # self.EMG_to_Activation_Mat = nn.Parameter(torch.eye(self.EMG_Channel_Count, self.muscle_num , dtype=torch.float, device=self.device)/self.EMG_mat_Lr)

    def bilinearInit(self):  
        from Dynamics2.Muscle_bilinear import Muscle

        # Muscles
        self.m1 = Muscle(100, 2000, 0.06, 0.006, [-0.05])
        self.m2 = Muscle(100, 2000, 0.06, 0.006, [0.05])
        self.m3 = Muscle(100, 2000, 0.06, 0.006, [-0.05])
        self.m4 = Muscle(100, 2000, 0.06, 0.006, [0.05])
        self.m5 = Muscle(100, 2000, 0.06, 0.006, [-0.05])
        self.m6 = Muscle(100, 2000, 0.06, 0.006, [0.05])
        self.m7 = Muscle(100, 2000, 0.06, 0.006, [-0.05])
        self.m8 = Muscle(100, 2000, 0.06, 0.006, [0.05])

        self.params = {'I': [0.004], 'K': 20, 'B': 2, 'K_': 40, 'B_': 4, 'speed_mode': False, 'K0_': 2000, 'K1_': 40000, 'L0_': 0.03, 'L1_': 0.006, 'I_': 0.008, 'M_': 0.05}

        self.numStates = 2
        
    def shadmehrInit(self):
        from Dynamics2.Muscle_Shadmehr import Muscle

        # lengths (muscle attachment geometry)
        # lengths = [[.1, .5], [.1, .5], [.1, .5], [.1, .5]]
        lengths = [[.1, .5]]
        L0s = np.linalg.norm(lengths, axis=1)

        # Muscle parameters: f_max, lM_opt, k_SE, k_PE, b, gamma
        fMax_def = 3000
        kSE_def = 1000
        kPE_def = kSE_def/10
        b_def = kSE_def/100
        gamma_def = 0.5

        # Joint parameters: B, K, I
        K_def = 20
        B_def = 1
        I_def = 1e-3

        # bicep-tricep
        self.m1 = Muscle(fMax_def, L0s[0], kSE_def, kPE_def, b_def, gamma_def)
        self.m2 = Muscle(fMax_def, L0s[0], kSE_def, kPE_def, b_def, gamma_def)
        
        # index flexion-extension
        self.m3 = Muscle(fMax_def, L0s[0], kSE_def, kPE_def, b_def, gamma_def)
        self.m4 = Muscle(fMax_def, L0s[0], kSE_def, kPE_def, b_def, gamma_def)
        
        # digits flexion-extension
        self.m5 = Muscle(fMax_def, L0s[0], kSE_def, kPE_def, b_def, gamma_def)
        self.m6 = Muscle(fMax_def, L0s[0], kSE_def, kPE_def, b_def, gamma_def)
        
        # thumb
        self.m7 = Muscle(fMax_def, L0s[0], kSE_def, kPE_def, b_def, gamma_def)
        self.m8 = Muscle(fMax_def, L0s[0], kSE_def, kPE_def, b_def, gamma_def)

        # underscores denote scaling on the learning rates - learn larger parameters faster to explore the space more
        self.params = {'I': I_def, 'K': K_def, 'B': B_def, 'L':lengths, 'K_': 20, 'B_': 1, 'I_': 1e-3, 'fMax_': 60000, 'lOpt_': 10.2, 'kSE_': 20000, 'kPE_': 2000, 'b_': 200, 'gamma_': 10}

        self.numStates = 4

        # for reference
        # self.Joint1 = Joint_1dof(self.device, self.AMI1, inertias=I_def, K=K_def, B=B_def, lengths=self.lengths[0], Lr_scale=Dynamic_Lr, NN_ratio=NN_ratio, \
        #                          K_scale=K_def, B_scale=B_def, I_scale=I_def, f_max_scale=fMax_def, lM_opt_scale=L0s[0], kSEs_scale=kSE_def, kPEs_scale=kPE_def, bs_scale=b_def, gamma_scale=gamma_def)
        
        # self.Joint2 = Joint_1dof(self.device, self.AMI2, inertias=I_def, K=K_def, B=B_def, lengths=self.lengths[1], Lr_scale=Dynamic_Lr, NN_ratio=NN_ratio, \
        #                          K_scale=K_def, B_scale=B_def, I_scale=I_def, f_max_scale=fMax_def, lM_opt_scale=L0s[1], kSEs_scale=kSE_def, kPEs_scale=kPE_def, bs_scale=b_def, gamma_scale=gamma_def)
        
        # self.Joint3 = Joint_1dof(self.device, self.AMI3, inertias=I_def, K=K_def, B=B_def, lengths=self.lengths[2], Lr_scale=Dynamic_Lr, NN_ratio=NN_ratio, \
        #                          K_scale=K_def, B_scale=B_def, I_scale=I_def, f_max_scale=fMax_def, lM_opt_scale=L0s[2], kSEs_scale=kSE_def, kPEs_scale=kPE_def, bs_scale=b_def, gamma_scale=gamma_def)
        
        # self.Joint4 = Joint_1dof(self.device, self.AMI4, inertias=I_def, K=K_def, B=B_def, lengths=self.lengths[3], Lr_scale=Dynamic_Lr, NN_ratio=NN_ratio, \
        #                          K_scale=K_def, B_scale=B_def, I_scale=I_def, f_max_scale=fMax_def, lM_opt_scale=L0s[3], kSEs_scale=kSE_def, kPEs_scale=kPE_def, bs_scale=b_def, gamma_scale=gamma_def)

    def thelenInit(self):
        from Dynamics2.Muscle_Thelen import Muscle

        # lengths = [[.5, 1], [.5, 1], [.5, 1], [.5, 1]]
        lengths = [[.1, .5]]

        # Muscle parameters: f_max, lM_opt, vM_max, lT_slack
        fMax_def = 3000
        lM_opt_def = 0.9*np.linalg.norm(lengths, axis=1)
        vM_max_def = 10
        lT_slack_def = 0.3*lM_opt_def

        # Joint parameters: B, K, I
        K_def = 20
        B_def = 1
        I_def = 1e-3

        # bicep-tricep
        self.m1 = Muscle(fMax_def, lM_opt_def, vM_max_def, lT_slack_def)
        self.m2 = Muscle(fMax_def, lM_opt_def, vM_max_def, lT_slack_def)
        
        # index flexion-extension
        self.m3 = Muscle(fMax_def, lM_opt_def, vM_max_def, lT_slack_def)
        self.m4 = Muscle(fMax_def, lM_opt_def, vM_max_def, lT_slack_def)
        
        # digits flexion-extension
        self.m5 = Muscle(fMax_def, lM_opt_def, vM_max_def, lT_slack_def)
        self.m6 = Muscle(fMax_def, lM_opt_def, vM_max_def, lT_slack_def)
        
        # thumb
        self.m7 = Muscle(fMax_def, lM_opt_def, vM_max_def, lT_slack_def)
        self.m8 = Muscle(fMax_def, lM_opt_def, vM_max_def, lT_slack_def)

        self.params = {'I': I_def, 'K': K_def, 'B': B_def, 'L':lengths, 'fMax': fMax_def, 'lOpt': lM_opt_def, 'vM': vM_max_def, 'lT': lT_slack_def}

        self.numStates = 4

    def forward(self, SS, EMG, dt):
        # Get the muscle activations then pass them into the joint model.
        Alphas = torch.matmul(EMG[:, 0:self.EMG_Channel_Count], self.EMG_to_Activation_Mat*self.EMG_mat_Lr)
        
        # print(Alphas)
        # print("Joint1 ---------------------------------------------------------------------------------")
        rw1, rs1 = self.Joint1(SS[:, 0*self.numStates:1*self.numStates], Alphas[:, 0:2], dt)
        # print("Joint2 ---------------------------------------------------------------------------------")
        rw2, rs2 = self.Joint2(SS[:, 1*self.numStates:2*self.numStates], Alphas[:, 2:4], dt)
        # print("Joint3 ---------------------------------------------------------------------------------")
        rw3, rs3 = self.Joint3(SS[:, 2*self.numStates:3*self.numStates], Alphas[:, 4:6], dt)
        # print("Joint4 ---------------------------------------------------------------------------------")
        rw4, rs4 = self.Joint4(SS[:, 3*self.numStates:4*self.numStates], Alphas[:, 6:8], dt)

        rw = torch.hstack([rw1, rw2, rw3, rw4])
        rs = torch.hstack([rs1, rs2, rs3, rs4])

        return rw, rs

    def lock_EMG_mat(self, switch=True):
        self.EMG_to_Activation_Mat.requires_grad = not switch
        
    def lock_I(self, switch=True):
        self.Joint1.I.requires_grad = not switch
        self.Joint2.I.requires_grad = not switch
        self.Joint3.I.requires_grad = not switch
        self.Joint4.I.requires_grad = not switch
        
    def lock_for_robot(self, switch=True):
        self.lock_EMG_mat(switch)
        self.lock_I(switch)
        
    def disable_NN(self):
        self.Joint1.disable_NN()
        self.Joint2.disable_NN()
        self.Joint3.disable_NN()
        self.Joint4.disable_NN()
        
    def enable_NN(self):
        self.Joint1.enable_NN()
        self.Joint2.enable_NN()
        self.Joint3.enable_NN()
        self.Joint4.enable_NN()
        
    def print_params(self):
        # print all parameters of the dynamic model
        self.Joint1.print_params()
        self.Joint2.print_params()
        self.Joint3.print_params()
        self.Joint4.print_params()
        print("EMG to Muscle Activation mat:\n", (self.EMG_to_Activation_Mat*self.EMG_mat_Lr).detach().cpu().numpy())
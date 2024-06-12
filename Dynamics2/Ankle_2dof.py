# BUGMAN Feb 1 2022
# The whole model for two degree of freedom ankle
"""
Moment arm from opensim
		Up/Down		In/Out
TA		0.04		0.0125
GAS		-0.045	    0.0025
TP		-0.01		0.02
PL(FL)	-0.01		-0.0275

Inertia from opensim
I_up_down,  kg*m2   0.008
I_in_out,   kg*m2   0.004

"""

from re import I
import torch
from torch import nn
from torch.functional import F

from Dynamics2.Muscle import Muscle
from Dynamics2.Joint_2dof_NN_multiply import Joint_2dof

# For Testing
# from Muscle import Muscle
# from Joint_2dof import Joint_2dof

class Ankle_2dof(nn.Module):
    def __init__(self, device, EMG_Channel_Count, Left, Dyanmic_Lr, EMG_mat_Lr, NN_ratio, \
                 K0_scale=2000, K1_scale=40000, L0_scale=0.03, L1_scale=0.006, I_scale=0.008, M_scale=0.05, speed_mode=False):
        super().__init__()
        
        self.device = device
        self.EMG_Channel_Count = EMG_Channel_Count
        self.NN_ratio = NN_ratio
        self.EMG_mat_Lr = EMG_mat_Lr
        
        # Muscles
        
        if(Left == True):
            self.TP = Muscle(2000, 40000, 0.03, 0.006, [0.02, -0.01])
            self.PL = Muscle(2000, 40000, 0.05, 0.006, [-0.0275, -0.01])
        else:
            self.TP = Muscle(2000, 40000, 0.03, 0.006, [-0.02, -0.01])
            self.PL = Muscle(2000, 40000, 0.05, 0.006, [0.0275, -0.01])
        self.TA  = Muscle(2000, 40000, 0.098, 0.006, [0.0125, 0.04])
        self.GAS = Muscle(2000, 40000, 0.05, 0.006, [0.0025, -0.045])
         
        self.muscles = [self.TP, self.PL, self.TA, self.GAS]
        
        # For testing
        # m1 = Muscle(2000, 40000, 0.03, 0.006, [0.05, 0])
        # m2 = Muscle(2000, 40000, 0.03, 0.006, [-0.05, 0])
        # m3 = Muscle(2000, 40000, 0.03, 0.006, [0, 0.05])
        # m4 = Muscle(2000, 40000, 0.03, 0.006, [0, -0.05])
        # self.muscles = [m1, m2, m3, m4]

        
        self.muscle_num = len(self.muscles)

        # Inertia
        self.I = [0.004, 0.008]
        
        # Joint
        self.Joint = Joint_2dof(self.device, self.muscles, inertias = self.I, Lr_scale = Dyanmic_Lr, NN_ratio=NN_ratio, \
                                K0_scale = K0_scale, K1_scale = K1_scale, L0_scale =  L0_scale, \
                                L1_scale = L1_scale, I_scale = I_scale, M_scale = M_scale, speed_mode = speed_mode)
        
        # EMG to Muscle activation matrix
        # There will be two initialization style of the EMG to muscle activation matrix.
        # If the electrodes are right upon the targeted muscle eye matrix should be chose, otherwise all-one matrix should be the way to go 
        # I'm using the scaled all-one matrix implementation here.
        # self.EMG_to_Activation_Mat = nn.Parameter(torch.ones((self.muscle_num, self.EMG_Channel_Count), dtype=torch.float, device=self.device)/self.EMG_mat_Lr/self.EMG_Channel_Count)
        self.EMG_to_Activation_Mat = nn.Parameter(torch.eye(self.muscle_num, self.EMG_Channel_Count, dtype=torch.float, device=self.device)/self.EMG_mat_Lr)
        
    def forward(self, SS, EMG, dt):
        # Get the muscle activations then pass them into the joint model.
        Alphas = torch.matmul(EMG, self.EMG_to_Activation_Mat*self.EMG_mat_Lr)
        # TODO: #3 Add nonlinear calculation to EMG
        # print(Alphas)
        return self.Joint(SS, Alphas, dt)

    def lock_EMG_mat(self, switch=True):
        self.EMG_to_Activation_Mat.requires_grad = not switch
        
    def lock_GAS(self, switch=True):
        self.Joint.K0s[3].requires_grad = not switch
        self.Joint.K1s[3].requires_grad = not switch
        self.Joint.L0s[3].requires_grad = not switch
        self.Joint.L1s[3].requires_grad = not switch
        
    def lock_I(self, switch=True):
        self.Joint.I.requires_grad = not switch
        
    def lock_for_robot(self, switch=True):
        self.lock_EMG_mat(switch)
        self.lock_GAS(switch)
        self.lock_I(switch)
        
    def disable_NN(self):
        self.Joint.disalbe_NN()
        
    def enable_NN(self):
        self.Joint.enable_NN()
        
    def print_params(self):
        # print all parameters of the dynamic model
        self.Joint.print_params()
        print("EMG to Muscle Activation mat:\n", (self.EMG_to_Activation_Mat*self.EMG_mat_Lr).detach().cpu().numpy())

# Test
"""
device = torch.device("cpu")
model = Ankle_2dof(device, 4, Left=True, Dyanmic_Lr=5, EMG_mat_Lr=20.0, NN_ratio=0.0)
model.eval()
X = torch.FloatTensor([[0,0,0,0],[0,0,0,0],[1,0,0,0]])
a1 = torch.FloatTensor([[0,0,0,0],[0.1,0,1,0],[0,0,1,0]])
for i in range(10):
    X = model(X,a1)
    print(X)
    # print(X[1,0])
"""
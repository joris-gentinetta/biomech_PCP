# BUGMAN Jan 29 2022
"""
This class is the parent class of all kinds of muscle pulling joints.
Rotational inertias for each axis, their reference values are: (values come from the opensim model2392)
I_up_down,  kg*m2   0.008
I_in_out,   kg*m2   0.004
"""
import torch
from torch import nn
import numpy as np

class Joint(nn.Module):
    def __init__(self, device, muscles, inertias, damping, stiffness, Lr_scale, \
                 K0_scale=2000, K1_scale=40000, L0_scale=0.07, L1_scale=0.006, I_scale=0.008, B_scale=4, K_scale=40, M_scale=0.05):
        super().__init__()
        
        self.device = device
        self.muscle_num = len(muscles)
        # print("muscle num", self.muscle_num)
        self.init_muscles = muscles
        self.init_inertias = inertias
        self.damping = damping
        self.stiffness = stiffness
        
        # Scales to assign custom learning rate on each parameter
        self.Lr_scale = Lr_scale
        self.I_scale = I_scale
        self.K0_scale = K0_scale
        self.K1_scale = K1_scale
        self.L0_scale = L0_scale
        self.L1_scale = L1_scale
        self.M_scale = M_scale # Muscle moment arm scale factor
        self.K_scale = K_scale
        self.B_scale = B_scale
        
        # Generate all the parameters
        self.K0s = nn.ParameterList([nn.Parameter(data=torch.tensor([m.K0/self.K0_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.K1s = nn.ParameterList([nn.Parameter(data=torch.tensor([m.K1/self.K1_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.L0s = nn.ParameterList([nn.Parameter(data=torch.tensor([m.L0/self.L0_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.L1s = nn.ParameterList([nn.Parameter(data=torch.tensor([m.L1/self.L1_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.Ms  = nn.ParameterList([nn.Parameter(data=torch.tensor(np.array(m.M/self.M_scale/self.Lr_scale), dtype=torch.float, device=self.device)) for m in self.init_muscles]) # Muscle moment arm [x,y]
        self.As  = nn.ParameterList([nn.Parameter(data=torch.tensor([m.A/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.I = nn.Parameter(data=torch.tensor(np.array(self.init_inertias)/self.I_scale/self.Lr_scale, dtype=torch.float, device=self.device))
        self.B = nn.Parameter(data=torch.tensor(np.array(self.damping)/self.B_scale/self.Lr_scale, dtype=torch.float, device=self.device))
        self.K = nn.Parameter(data=torch.tensor(np.array(self.stiffness)/self.K_scale/self.Lr_scale, dtype=torch.float, device=self.device))
        
    def forward(self, SS, A, dt = 0.0166667, speed_mod=False):
        pass
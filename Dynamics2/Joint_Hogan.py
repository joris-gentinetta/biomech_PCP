# Mikey Fernandez 07/19/2023
"""

"""

import torch
from torch import nn
import numpy as np

class Joint(nn.Module):
    def __init__(self, device, muscles, inertias, damping, Lr_scale, I_scale=0.008, B_scale=4, K_scale=40, T_scale=100):
        super().__init__()
        
        self.device = device
        self.muscle_num = len(muscles)
        self.init_muscles = muscles
        self.init_inertias = inertias
        self.damping = damping
        
        # Scales to assign custom learning rate on each parameter
        self.Lr_scale = Lr_scale
        self.I_scale = I_scale
        self.T_scale = T_scale
        self.K_scale = K_scale
        self.B_scale = B_scale
        
        # Generate all the parameters
        self.Ts = nn.ParameterList([nn.Parameter(data=torch.tensor([m.T/self.T_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.Ks = nn.ParameterList([nn.Parameter(data=torch.tensor([m.K/self.K_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.I = nn.Parameter(data=torch.tensor(np.array(self.init_inertias)/self.I_scale/self.Lr_scale, dtype=torch.float, device=self.device))
        self.B = nn.Parameter(data=torch.tensor(np.array(self.damping)/self.B_scale/self.Lr_scale, dtype=torch.float, device=self.device))
        
    def forward(self, SS, A, dt=0.0166667):
        pass
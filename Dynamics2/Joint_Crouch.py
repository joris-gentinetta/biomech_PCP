# BUGMAN Jan 29 2022
# Modified by Mikey Fernandez 07/19/2022

"""
This class implements a Crouch Hill model as in https://reader.elsevier.com/reader/sd/pii/S0021929016311368?token=C970342A709318CD89AC3B9A66584EC328EE9512B653454568141A91DC6D12F625C30EA8311D0E646A1D19EB8C337F54&originRegion=us-east-1&originCreation=20220718220756
"""

import torch
from torch import nn
import numpy as np

class Joint(nn.Module):
    def __init__(self, device, muscles, inertias, K, B, lengths, Lr_scale, \
                K_scale=1, B_scale=1, I_scale=1, f_max_scale=1, lM_opt_scale=1, kPEs_scale=1, vMaxs_scale=1, W_scale=1, c_scale=1):
        super().__init__()
        
        self.device = device
        self.muscle_num = len(muscles)
        self.init_muscles = muscles
        self.init_inertias = inertias

        self.L2 = lengths[0][0]
        self.L3 = lengths[0][1]
        
        # Scales to assign custom learning rate on each parameter (muscles)
        self.Lr_scale = Lr_scale
        self.f_max_scale = f_max_scale
        self.lM_opt_scale = lM_opt_scale
        self.kPEs_scale = kPEs_scale
        self.vMaxs_scale = vMaxs_scale
        self.W_scale = W_scale
        self.c_scale = c_scale

        self.I_scale = I_scale
        self.K_scale = K_scale
        self.B_scale = B_scale

        # Generate all the muscle parameters
        self.f_maxs = nn.ParameterList([nn.Parameter(data=torch.tensor([m.f_max/self.f_max_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.lM_opts = nn.ParameterList([nn.Parameter(data=torch.tensor([m.lM_opt/self.lM_opt_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.kPEs = nn.ParameterList([nn.Parameter(data=torch.tensor([m.k_PE/self.kPEs_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.vMaxs = nn.ParameterList([nn.Parameter(data=torch.tensor([m.v_max/self.vMaxs_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.Ws = nn.ParameterList([nn.Parameter(data=torch.tensor([m.W/self.W_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.cs = nn.ParameterList([nn.Parameter(data=torch.tensor([m.c/self.c_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])

        # Generate the joint parameters
        self.I = nn.Parameter(data=torch.tensor(np.array(self.init_inertias)/self.I_scale/self.Lr_scale, dtype=torch.float, device=self.device))
        self.K = nn.Parameter(data=torch.tensor(K/self.K_scale/self.Lr_scale, dtype=torch.float, device=self.device))
        self.B = nn.Parameter(data=torch.tensor(B/self.B_scale/self.Lr_scale, dtype=torch.float, device=self.device))

    def forward(self, SS, dt = 0.0166667, speed_mod=False):
        pass

    def lock_params(self, state=True):
        for i in range(self.muscle_num):
            self.f_maxs[i].requires_grad = not state
            self.lM_opts[i].requires_grad = not state
            self.kPEs[i].requires_grad = not state
            self.vMaxs[i].requires_grad = not state
            self.Ws[i].requires_grad = not state
            self.cs[i].requires_grad = not state

        self.I.requires_grad = not state
        self.K.requires_grad = not state
        self.B.requires_grad = not state
# BUGMAN Jan 29 2022
# Modified by Mikey Fernandez 06/06/2022

"""
This class is the parent class for all joint models based on the Shadmehr MTU
"""

import torch
from torch import nn
from torch.functional import F
import numpy as np

class Joint(nn.Module):
    def __init__(self, device, muscles, inertias, K, B, lengths, Lr_scale, \
                K_scale=1, B_scale=1, I_scale=1, f_max_scale=1, lM_opt_scale=1, kSEs_scale=1, kPEs_scale=1, bs_scale=1, gamma_scale=1):
        super().__init__()
        
        self.device = device
        self.muscle_num = len(muscles)
        # print("muscle num", self.muscle_num)
        self.init_muscles = muscles
        self.init_inertias = inertias

        self.L2 = lengths[0][0]
        self.L3 = lengths[0][1]
        
        # Scales to assign custom learning rate on each parameter (muscles)
        self.Lr_scale = Lr_scale
        self.f_max_scale = f_max_scale
        self.lM_opt_scale = lM_opt_scale
        self.kSEs_scale = kSEs_scale
        self.kPEs_scale = kPEs_scale
        self.bs_scale = bs_scale
        self.gamma_scale = gamma_scale

        self.I_scale = I_scale
        self.K_scale = K_scale
        self.B_scale = B_scale

        # Generate all the muscle parameters
        self.f_maxs = nn.ParameterList([nn.Parameter(data=torch.tensor([m.f_max/self.f_max_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.lM_opts = nn.ParameterList([nn.Parameter(data=torch.tensor([m.lM_opt/self.lM_opt_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.kSEs = nn.ParameterList([nn.Parameter(data=torch.tensor([m.k_SE/self.kSEs_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.kPEs = nn.ParameterList([nn.Parameter(data=torch.tensor([m.k_PE/self.kPEs_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.bs = nn.ParameterList([nn.Parameter(data=torch.tensor([m.b/self.bs_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.gammas = nn.ParameterList([nn.Parameter(data=torch.tensor([m.gamma/self.gamma_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])

        # Generate the joint parameters
        self.I = nn.Parameter(data=torch.tensor(np.array(self.init_inertias)/self.I_scale/self.Lr_scale, dtype=torch.float, device=self.device))
        self.K = nn.Parameter(data=torch.tensor(K/self.K_scale/self.Lr_scale, dtype=torch.float, device=self.device))
        self.B = nn.Parameter(data=torch.tensor(B/self.B_scale/self.Lr_scale, dtype=torch.float, device=self.device))

    def forward(self, SS, Alphas, dt=0.0166667):
        pass

    def lock_params(self, state=True):
        for i in range(self.muscle_num):
            self.f_maxs[i].requires_grad = not state
            self.lM_opts[i].requires_grad = not state
            self.kSEs[i].requires_grad = not state
            self.kPEs[i].requires_grad = not state
            self.bs[i].requires_grad = not state
            self.gammas[i].requires_grad = not state

        self.I.requires_grad = not state
        self.K.requires_grad = not state
        self.B.requires_grad = not state
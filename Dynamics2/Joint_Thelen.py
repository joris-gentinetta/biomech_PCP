# BUGMAN Jan 29 2022
# Modified by Mikey Fernandez 04/02/2022

"""
This class is the parent class for all joint models based on the Thelen MTU
"""

import torch
from torch import nn
from torch.functional import F
import numpy as np

class Joint(nn.Module):
    def __init__(self, device, muscles, inertias, damping, stiffness, Lr_scale, \
                f_max_=1, lM_opt_=1, vM_max_=1, vAlpha_=1, lT_slack_=1, pen_=1, moment_arm_=1, I_=1, K_=1, B_=1, lIn_=1, lOrig_=1):
        super().__init__()
        
        self.device = device
        self.muscle_num = len(muscles)
        self.init_muscles = muscles
        self.init_inertias = inertias
        self.damping = damping
        self.stiffness = stiffness
        
        # Scales to assign custom learning rate on each parameter
        self.Lr_scale = Lr_scale
        self.f_max_scale = f_max_
        self.lM_opt_scale = lM_opt_
        self.vM_max_scale = vM_max_
        self.vAlpha_scale = vAlpha_
        self.lT_slack_scale = lT_slack_
        self.pen_scale = pen_
        # self.moment_arm_scale = moment_arm_scale
        self.lIn_scale = lIn_
        self.lOrig_scale = lOrig_

        self.I_scale = I_
        self.K_scale = K_
        self.B_scale = B_
        # self.M_scale = M_scale

        # constraining
        self.minActivation = 0.01

        # Generate all the parameters
        self.f_maxs = nn.ParameterList([nn.Parameter(data=torch.tensor([m.f_max/self.f_max_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.lM_opts = nn.ParameterList([nn.Parameter(data=torch.tensor([m.lM_opt/self.lM_opt_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.vM_maxs = nn.ParameterList([nn.Parameter(data=torch.tensor([m.vM_max/self.vM_max_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.vAlphas = nn.ParameterList([nn.Parameter(data=torch.tensor([m.vAlpha/self.vAlpha_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.lT_slacks = nn.ParameterList([nn.Parameter(data=torch.tensor([m.lT_slack/self.lT_slack_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.pen_opts = nn.ParameterList([nn.Parameter(data=torch.tensor([m.pen_opt/self.pen_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        # self.moment_arms  = nn.ParameterList([nn.Parameter(data=torch.tensor(np.array(m.moment_arm/self.moment_arm_scale/self.Lr_scale), dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.lIns = nn.ParameterList([nn.Parameter(data=torch.tensor([m.lIn/self.lIn_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])
        self.lOrigs = nn.ParameterList([nn.Parameter(data=torch.tensor([m.lOrig/self.lOrig_scale/self.Lr_scale], dtype=torch.float, device=self.device)) for m in self.init_muscles])

        self.I = nn.Parameter(data=torch.tensor(np.array(self.init_inertias)/self.I_scale/self.Lr_scale, dtype=torch.float, device=self.device))
        self.K = nn.Parameter(data=torch.tensor(np.array(self.stiffness)/self.K_scale/self.Lr_scale, dtype=torch.float, device=self.device))
        self.B = nn.Parameter(data=torch.tensor(np.array(self.damping)/self.B_scale/self.Lr_scale, dtype=torch.float, device=self.device))

    def forward(self, SS, A, dt = 0.0166667, speed_mod=False):
        pass
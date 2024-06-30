# BUGMAN Feb 17 2022
"""
The joint with only 1 degree of freedom
In this model, the total force is the sum of bilinear model force and neural network force
"""

import torch
from torch import nn
import numpy as np

class BilinearJoints(nn.Module):
    def __init__(self, device, muscles, parameters, dt, speed_mode=False):
        super().__init__()

        self.device = device
        self.muscle_num = len(muscles[0])
        self.joint_num = len(muscles)
        self.dt = dt

        self.K0s = nn.Parameter(data=torch.tensor([[m.K0 for m in joint] for joint in muscles], dtype=torch.float, device=self.device))
        self.K1s = nn.Parameter(data=torch.tensor([[m.K1 for m in joint] for joint in muscles], dtype=torch.float, device=self.device))
        self.L0s = nn.Parameter(data=torch.tensor([[m.L0 for m in joint] for joint in muscles], dtype=torch.float, device=self.device))
        self.L1s = nn.Parameter(data=torch.tensor([[m.L1 for m in joint] for joint in muscles], dtype=torch.float, device=self.device))
        self.Ms = nn.Parameter(data=torch.tensor([[m.M for m in joint] for joint in muscles], dtype=torch.float, device=self.device))
        self.As = nn.Parameter(data=torch.tensor([[m.A for m in joint] for joint in muscles], dtype=torch.float, device=self.device))
        self.I = nn.Parameter(data=torch.tensor(parameters['I'], dtype=torch.float, device=self.device).repeat(self.joint_num))
        self.B = nn.Parameter(data=torch.tensor(parameters['B'], dtype=torch.float, device=self.device).repeat(self.joint_num))
        self.K = nn.Parameter(data=torch.tensor(parameters['K'], dtype=torch.float, device=self.device).repeat(self.joint_num))

        self.speed_mode = speed_mode

    def forward(self, SS, Alphas):
        """Calculate the Joint dynamic for one step
        Output the new system state

        Args:
            SS (torch.tensor): System states. They are [wx, wy, dwx, dwy]. 
            Alphas (torch.tensor): Muscle activations, [batch_size * muscle_number]
            self.dt (float, optional): Delta t between each iteration. Defaults to 0.0166667.
        """
        batch_size = len(Alphas)
        
        #################################
        # Calculate the offset          #
        #################################
        """
        Because of the property of the mechanical model used here,
        the positions in the system state will not be zero when there are no muscle activations.
        So a offset is needed here to make sure cursor to be at center.
        The offset is calculated based on the K0s, L0s and moment arms.
        """
        # BB = torch.tensor([0, 0], dtype = torch.float, device=self.device)
        # AA = torch.zeros((2, 2), dtype = torch.float, device=self.device)
        # for i in range(self.muscle_num):
        #     BB += K0s[i]*Ms[i]*L0s[i]
        #     AA += K0s[i]*torch.matmul(Ms[i].view(-1,1), Ms[i].view(1,-1))
        # offset = -torch.linalg.solve(AA,BB)
        # # Since the offset will always
        # offset = offset.detach()

        # Alpha is the clamped muscle activation. It should between 0 to 1
        # Alpha's shape is (batch_size, muscle_number)

        Alphas = torch.clamp(Alphas, 0, 1)

        muscle_SSs = SS.unsqueeze(3) * self.Ms.unsqueeze(0).unsqueeze(2) # [batch_size, joint_num, state_num (2), muscle_num (2)]
        # Compute the dynamic model
        """
        System state simulation
        xdot = A x + B u
        # Linear interpolation between steps
        # Algorithm: to integrate from time 0 to time self.dt, with linear
        # interpolation between inputs u(0) = u0 and u(self.dt) = u1, we solve
        #   xdot = A x + B u,        x(0) = x0
        #   udot = (u1 - u0) / self.dt,   u(0) = u0.
        #
        # Solution is
        #   [ x(self.dt) ]       [ A*self.dt  B*self.dt  0 ] [  x0   ]
        #   [ u(self.dt) ] = exp [  0     0    I ] [  u0   ]
        #   [u1 - u0]       [  0     0    0 ] [u1 - u0]
        
        In this case,
        A = [[0, 1], [-(k1+k2)/I, -b/I]]
        B = [[0, 0], [(k2*L2-k1*L1)/I, 1/I]]
        X = [Position, Speed]
        Args:
            X : System States [Position, Speed]
            a1, a2 ([type]): The activation level of the two muscle.
            self.dt : Time between this frame to last frame
        """

        #################
        #   Matrix A    #
        #################
        A00 = torch.zeros((batch_size, self.joint_num), dtype=torch.float, device=self.device)  # [batch_size, joint_num]
        A01 = torch.ones((batch_size, self.joint_num), dtype=torch.float, device=self.device)

        K = self.K0s.unsqueeze(0) + self.K1s.unsqueeze(0) * Alphas
        K = K * self.Ms.unsqueeze(0) * self.Ms.unsqueeze(0)  # [batch_size, joint_num, muscle_num]
        K = K.sum(dim=2) # sum over muscles - > [batch_size, joint_num]
        A10 = -(K + self.K.unsqueeze(0))/self.I.unsqueeze(0) # [batch_size, joint_num]

            #############################
            #   DAMPING                 #
            #############################
        D = torch.sqrt(K * self.I.unsqueeze(0)) * 2
        A11 = -(D + self.B.unsqueeze(0)) / self.I.unsqueeze(0)  # with joint damping

        A = torch.stack([torch.stack([A00, A01], dim=2), torch.stack([A10, A11], dim=2)], dim=2)  # [batch_size, joint_num, 2, 2]


        #########################
        #   MATRIX B            #
        #########################
        B0 = torch.zeros(batch_size, self.joint_num, 2, dtype=torch.float, device=self.device)

        # The total force from one muscle (if the muscle is not stretched)
        B_F = ((self.K0s.unsqueeze(0) + self.K1s.unsqueeze(0) * Alphas)
               * (self.L0s.unsqueeze(0) + self.L1s.unsqueeze(0) * Alphas - torch.abs(muscle_SSs[:, :, 0, :]))
               + self.K1s.unsqueeze(0) * self.L1s.unsqueeze(0) * Alphas * Alphas)

        # The following K is respect to w(angle)
        B10 = B_F * self.Ms.unsqueeze(0) / self.I.unsqueeze(0).unsqueeze(2)
        B10 = B10.sum(dim=2)
        B11 = torch.ones((batch_size, self.joint_num), dtype=torch.float, device=self.device) / self.I.unsqueeze(0)
        B1 = torch.stack([B10, B11], dim=2)

        B = torch.stack([B0, B1], dim=2)


        #########################
        #   U (1,1,Tx,Ty)       #
        #########################
        U0 = torch.zeros(batch_size, self.joint_num, 2, dtype=torch.float, device=self.device)
        U0[:, :, 0] = 1
        U1 = U0.clone()  # todo

        if not self.speed_mode:
            #############################
            #   Accurate Simulation     #
            #############################
            M = torch.cat([torch.cat([A*self.dt, B*self.dt, torch.zeros((batch_size, self.joint_num, 2, 2), dtype=torch.float, device=self.device)], dim=3),
                              torch.cat([torch.zeros((batch_size, self.joint_num, 2, 4), dtype=torch.float, device=self.device),  torch.eye(2).unsqueeze(0).repeat(batch_size, self.joint_num, 1, 1)], dim=3),
                              torch.zeros((batch_size, self.joint_num, 2, 6), dtype=torch.float, device=self.device)], dim=2)

            expMT = torch.matrix_exp(M)
            Ad = expMT[:, :, :2, :2]
            Bd1 = expMT[:, :, :2, 4:]
            Bd0 = expMT[:, :, :2, 2:4] - Bd1

            SSout = (torch.matmul(Ad, SS.unsqueeze(3)) + torch.matmul(Bd0, U0.unsqueeze(3)) + torch.matmul(Bd1, U1.unsqueeze(3))).squeeze(3)
        else:
            #############################
            #   Simplified Simulation   #
            #############################
            # The simplified simulation addition instead of integration
            # xdot = A x + B u
            SSout = (torch.matmul(A*self.dt, SS.unsqueeze(3)) + torch.matmul(B*self.dt, U0.unsqueeze(3)) + SS.unsqueeze(3)).squeeze(3)

        return SSout[:, :, 0], SSout

    def print_params(self):
        print(f'K0s: {self.K0s}')
        print(f'K1s: {self.K1s}')
        print(f'L0s: {self.L0s}')
        print(f'L1s: {self.L1s}')
        print(f'Moment arms: {self.Ms}')
        print(f'I: {self.I}')
        print(f'B: {self.B}')
        print(f'K: {self.K}')

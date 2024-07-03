# BUGMAN Feb 17 2022
"""
The joint with only 1 degree of freedom
In this model, the total force is the sum of bilinear model force and neural network force
"""
import math
import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization
from utils import Exponential


## Muscle parameters
K0 = math.log(100)
K1 = math.log(2000)
L0 = math.log(0.06)
L1 = math.log(0.006)

## Joint parameters
M = math.log(0.05)
I = math.log(0.004)
B = math.log(.3)
K = math.log(5)



class Muscles(nn.Module):
    def __init__(self, device, n_joints):
        super().__init__()

        self.device = device

        self.K0 = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * K0)
        self.K1 = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * K1)
        self.L0 = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * L0)
        self.L1 = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * L1)


        parameterization = Exponential()
        for parameter in ['K0', 'K1', 'L0', 'L1']:
            register_parametrization(self, parameter, parameterization)

    def forward(self, alphas, muscle_SS):
        alphas = torch.clamp(alphas, 0, 1)

        F = ((self.K0.unsqueeze(0) + self.K1.unsqueeze(0) * alphas)
               * (self.L0.unsqueeze(0) + self.L1.unsqueeze(0) * alphas - torch.abs(muscle_SS[:, :, 0, :])))  # todo abs

        K = self.K0.unsqueeze(0) + self.K1.unsqueeze(0) * alphas
        return F, K


class Joints(nn.Module):
    def __init__(self, device, n_joints, dt, speed_mode=False):
        super().__init__()

        self.device = device
        self.n_joints = n_joints
        self.dt = dt
        self.speed_mode = speed_mode


        self.I = nn.Parameter(data=torch.ones(self.n_joints, dtype=torch.float, device=self.device) * I)
        self.B = nn.Parameter(data=torch.ones(self.n_joints, dtype=torch.float, device=self.device) * B)
        self.K = nn.Parameter(data=torch.ones(self.n_joints, dtype=torch.float, device=self.device) * K)

        moment_arms = torch.ones((self.n_joints, 2), dtype=torch.float, device=self.device) * M
        moment_arms[:, 1] = moment_arms[:, 1] * -1
        self.M = nn.Parameter(data=moment_arms)

        parameterization = Exponential()
        for parameter in ['I', 'B', 'K']:
            register_parametrization(self, parameter, parameterization)

    def forward(self, F, K, SS):
        """Calculate the Joint dynamic for one step
        Output the new system state

        Args:
            SS (torch.tensor): System states. They are [wx, wy, dwx, dwy]. 
            Alphas (torch.tensor): Muscle activations, [batch_size * muscle_number]
            self.dt (float, optional): Delta t between each iteration. Defaults to 0.0166667.
        """

        batch_size = len(F)  # todo

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
        A00 = torch.zeros((batch_size, self.n_joints), dtype=torch.float, device=self.device)  # [batch_size, n_joints]
        A01 = torch.ones((batch_size, self.n_joints), dtype=torch.float, device=self.device)

        K = K * self.M.unsqueeze(0) * self.M.unsqueeze(0)  # [batch_size, n_joints, muscle_num]
        K = K.sum(dim=2) # sum over muscles - > [batch_size, n_joints]
        A10 = -(K + self.K.unsqueeze(0))/self.I.unsqueeze(0)  # [batch_size, n_joints] # todo

            #############################
            #   DAMPING                 #
            #############################

        D = torch.sqrt(K * self.I.unsqueeze(0)) * 2
        A11 = -(D + self.B.unsqueeze(0)) / self.I.unsqueeze(0)  # with joint damping

        A = torch.stack([torch.stack([A00, A01], dim=2), torch.stack([A10, A11], dim=2)], dim=2)  # [batch_size, n_joints, 2, 2]


        #########################
        #   MATRIX B            #
        #########################
        B0 = torch.zeros(batch_size, self.n_joints, 2, dtype=torch.float, device=self.device)

        B_F = F

        # The following K is respect to w(angle)
        B10 = B_F * self.M.unsqueeze(0) / self.I.unsqueeze(0).unsqueeze(2)
        B10 = B10.sum(dim=2)
        B11 = torch.ones((batch_size, self.n_joints), dtype=torch.float, device=self.device) / self.I.unsqueeze(0)
        B1 = torch.stack([B10, B11], dim=2)

        B = torch.stack([B0, B1], dim=2)


        #########################
        #   U (1,1,Tx,Ty)       #
        #########################
        U0 = torch.zeros(batch_size, self.n_joints, 2, dtype=torch.float, device=self.device)
        U0[:, :, 0] = 1
        U1 = U0 # todo

        if not self.speed_mode:
            #############################
            #   Accurate Simulation     #
            #############################
            M = torch.cat([torch.cat([A*self.dt, B*self.dt, torch.zeros((batch_size, self.n_joints, 2, 2), dtype=torch.float, device=self.device)], dim=3),
                              torch.cat([torch.zeros((batch_size, self.n_joints, 2, 4), dtype=torch.float, device=self.device),  torch.eye(2, dtype=torch.float, device=self.device).unsqueeze(0).repeat(batch_size, self.n_joints, 1, 1)], dim=3),
                              torch.zeros((batch_size, self.n_joints, 2, 6), dtype=torch.float, device=self.device)], dim=2)

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

        muscle_SSs = SSout.unsqueeze(3) * self.M.unsqueeze(0).unsqueeze(2) # [batch_size, n_joints, state_num (2), muscle_num (2)]

        return SSout[:, :, 0], muscle_SSs, SSout



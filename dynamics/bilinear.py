# BUGMAN Feb 17 2022
"""
The joint with only 1 degree of freedom
In this model, the total force is the sum of bilinear model force and neural network force
"""

import math
import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization
from dynamics.utils import Exponential, Sigmoid


## Muscle parameters
K0 = math.log(100)
K1 = math.log(2000)
L0 = math.log(0.06)
L1 = math.log(0.006)
B0 = math.log(math.e)
B1 = math.log(math.e)

## Joint parameters
M = 0.05
I = math.log(0.004)
B = math.log(math.e)
K = math.log(1)


class Muscles(nn.Module):
    def __init__(self, device, n_joints):
        super().__init__()

        self.device = device

        self.K0 = nn.Parameter(
            data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * K0
        )
        self.K1 = nn.Parameter(
            data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * K1
        )
        self.L0 = nn.Parameter(
            data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * L0
        )
        self.L1 = nn.Parameter(
            data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * L1
        )
        self.B0 = nn.Parameter(
            data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * B0
        )
        self.B1 = nn.Parameter(
            data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * B1
        )

        parameterization = Exponential()
        for parameter in ["K0", "K1", "L0", "L1", "B0", "B1"]:
            register_parametrization(self, parameter, parameterization)

    def forward(self, alphas, muscle_SS):
        alphas = torch.clamp(alphas, 0, 1)

        F = (self.K0.unsqueeze(0) + self.K1.unsqueeze(0) * alphas) * (
            self.L0.unsqueeze(0) + self.L1.unsqueeze(0) * alphas - muscle_SS[:, :, 0, :]
        ) + self.B0.unsqueeze(0) * torch.abs(
            muscle_SS[:, :, 1, :]
        ) ** self.B1.unsqueeze(0) * torch.sign(muscle_SS[:, :, 1, :])

        F = nn.functional.relu(F)  # wash out negative force!

        K = self.K0.unsqueeze(0) + self.K1.unsqueeze(0) * alphas

        return F, K


class Joints(nn.Module):
    def __init__(self, device, n_joints, dt, speed_mode=False):
        super().__init__()

        self.device = device
        self.n_joints = n_joints
        self.dt = dt
        self.speed_mode = speed_mode

        self.I = nn.Parameter(
            data=torch.ones(self.n_joints, dtype=torch.float, device=self.device) * I
        )
        self.B = nn.Parameter(
            data=torch.ones(self.n_joints, dtype=torch.float, device=self.device) * B
        )
        self.K = nn.Parameter(
            data=torch.ones(self.n_joints, dtype=torch.float, device=self.device) * K
        )

        moment_arms = (
            torch.ones((self.n_joints, 2), dtype=torch.float, device=self.device) * M
        )
        moment_arms[:, 0] = moment_arms[:, 0] * -1  # note that sign flipped
        self.M = nn.Parameter(data=moment_arms)

        parameterization = Exponential()
        # for parameter in ['I', 'B', 'K']:
        for parameter in ["I", "K"]:
            register_parametrization(self, parameter, parameterization)

        parameterization = Sigmoid()
        register_parametrization(self, "B", parameterization)

    def forward(self, F, K_musc, SS):
        """Calculate the joint dynamic for one step and output the new system state

        Args:
            F (torch.tensor): Muscle linear forces. They are [F0, F1]
            K_musc (torch.tensor): Muscle linear stiffness. They are [K0, K1]
            SS (torch.tensor): System states. They are [th, dth_dt].
        """

        batch_size = len(F)
        """
        System state simulation
        xdot = A x + B u
        # Linear interpolation between steps
        # Algorithm: to integrate from time 0 to time self.dt, with linear
        # interpolation between inputs u = [[F0], [F1]]
        
        #   xdot = A x + B u,        x(0) = x0
        
        # Solution is
        #   x(dt) = exp(A*dt) x0 + (I - exp(A*dt)) A^-1 B u

        In this case,
        A = [[0, 1], [-(K0+K1)/I, -b/I]]
        B = [[0, 0], [      M0/I, M1/I]] # note that the indices represent the values from different muscles, as appropriate
        X = [Position, Speed]
        U = [F0, F1]
        Args:
            X : System States [Position, Speed]
            F0, F1 (t): The activation level of the two muscle.
            self.dt : Timestep of integration
        """

        #################
        #   Matrix A    #
        #################
        A00 = torch.zeros(
            (batch_size, self.n_joints), dtype=torch.float, device=self.device
        )  # [batch_size, n_joints]
        A01 = torch.ones(
            (batch_size, self.n_joints), dtype=torch.float, device=self.device
        )

        K = (
            K_musc * self.M.unsqueeze(0) * self.M.unsqueeze(0)
        )  # [batch_size, n_joints, muscle_num]
        K = K.sum(dim=2) + self.K.unsqueeze(
            0
        )  # sum over muscles - > [batch_size, n_joints]
        A10 = -K / self.I.unsqueeze(0)  # [batch_size, n_joints]
        # A10 = -K / self.I.unsqueeze(0)  # [batch_size, n_joints]

        #############################
        #   DAMPING                 #
        #############################
        D = torch.sqrt(K * self.I.unsqueeze(0)) * 2
        A11 = -D * self.B.unsqueeze(0) / self.I.unsqueeze(0)  # with joint damping
        # A11 = -D / self.I.unsqueeze(0)  # with joint damping

        A = torch.stack(
            [torch.stack([A00, A01], dim=2), torch.stack([A10, A11], dim=2)], dim=2
        )  # [batch_size, n_joints, 2, 2]

        #########################
        #   MATRIX B            #
        #########################
        B0 = torch.zeros(
            batch_size, self.n_joints, 1, dtype=torch.float, device=self.device
        )

        B_F = F

        # The following K is respect to w(angle)
        B1 = (
            -B_F
            * self.M.unsqueeze(0)
            / self.I.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1)
        )
        B1 = B1.sum(dim=2).unsqueeze(2)  # sum over muscles - > [batch_size, n_joints]

        B = torch.stack([B0, B1], dim=2)

        if not self.speed_mode:
            #############################
            #   Accurate Simulation     #
            #############################

            expAt = torch.matrix_exp(A * self.dt)
            eye = (
                torch.eye(2, dtype=torch.float, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, self.n_joints, 1, 1)
            )

            SSout = torch.matmul(expAt, SS.unsqueeze(3)) + torch.matmul(
                torch.linalg.solve(A, expAt - eye), B
            )
            SSout = SSout.squeeze(3)

        else:
            #############################
            #   Simplified Simulation   #
            #############################
            # The simplified simulation addition instead of integration
            # xdot = A x + B u
            SSout = (
                (torch.matmul(A, SS.unsqueeze(3)) + B) * self.dt + SS.unsqueeze(3)
            ).squeeze(3)

        muscle_SSs = SSout.unsqueeze(3) * self.M.unsqueeze(0).unsqueeze(
            2
        )  # [batch_size, n_joints, state_num (2), muscle_num (2)]

        return (
            SSout[:, :, 0],
            muscle_SSs,
            SSout,
        )  # position, muscle state, [position, velocity]

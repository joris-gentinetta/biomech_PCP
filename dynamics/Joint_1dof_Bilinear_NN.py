# BUGMAN Feb 17 2022
"""
The joint with only 1 degree of freedom
In this model, the total force is the sum of bilinear model force and neural network force
"""

import torch
from torch import nn
import numpy as np

class Joint_1dof(nn.Module):
    def __init__(self, device, muscles, parameters, NN_ratio, dt):
        super().__init__()

        self.device = device
        self.muscle_num = len(muscles)
        self.dt = dt

        self.K0s = nn.Parameter(data=torch.tensor([m.K0 for m in muscles], dtype=torch.float, device=self.device))
        self.K1s = nn.Parameter(data=torch.tensor([m.K1 for m in muscles], dtype=torch.float, device=self.device))
        self.L0s = nn.Parameter(data=torch.tensor([m.L0 for m in muscles], dtype=torch.float, device=self.device))
        self.L1s = nn.Parameter(data=torch.tensor([m.L1 for m in muscles], dtype=torch.float, device=self.device))
        self.Ms = nn.Parameter(data=torch.tensor([m.M for m in muscles], dtype=torch.float, device=self.device))
        self.As = nn.Parameter(data=torch.tensor([m.A for m in muscles], dtype=torch.float, device=self.device))
        self.I = nn.Parameter(data=torch.tensor(np.array(parameters['I']), dtype=torch.float, device=self.device))
        self.B = nn.Parameter(data=torch.tensor(np.array(parameters['B']), dtype=torch.float, device=self.device))
        self.K = nn.Parameter(data=torch.tensor(np.array(parameters['K']), dtype=torch.float, device=self.device))

        # For the compensational_nn
        self.compensational_nns = nn.ModuleList([compensational_nn(device=self.device) for i in range(self.muscle_num)])
        self.speed_mode = parameters['speed_mode']
        self.designed_NN_ratio = NN_ratio
        self.NN_ratio = NN_ratio

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

        muscle_SSs = SS.unsqueeze(2) * self.Ms.unsqueeze(0).unsqueeze(0) # [batch_size, 2, muscle_num]
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
        # For matrix A: A00, A01, A10, A11 are all 2x2 matrix
        # System states are [Wx, Wy, dWx_dt, dWy_dt]
        # A00 = torch.tensor(np.array([[0,0],[0,0]]*batch_size), dtype=torch.float, device=self.device)
        A00 = torch.zeros(batch_size, dtype=torch.float, device=self.device)
        A01 = torch.ones(batch_size, dtype=torch.float, device=self.device)

        K = self.K0s.unsqueeze(0) + self.K1s.unsqueeze(0) * Alphas
        K = K * self.Ms.unsqueeze(0) * self.Ms.unsqueeze(0)
        K = K.sum(dim=1)
        A10 = -(K + self.K)/self.I # with joint stiffness

            #############################
            #   DAMPING                 #
            #############################
        D = torch.sqrt(K * self.I) * 2
        A11 = -(D + self.B) / self.I  # with joint damping

        A = torch.stack([torch.stack([A00, A01], dim=1), torch.stack([A10, A11], dim=1)], dim=1)


        #########################
        #   MATRIX B            #
        #########################
        B0 = torch.zeros(batch_size, 2, dtype=torch.float, device=self.device)

        # The total force from one muscle (if the muscle is not stretched)
        B_F = ((self.K0s.unsqueeze(0) + self.K1s.unsqueeze(0) * Alphas)
               * (self.L0s.unsqueeze(0) + self.L1s.unsqueeze(0) * Alphas - torch.abs(muscle_SSs[:, 0, :]))
               + self.K1s.unsqueeze(0) * self.L1s.unsqueeze(0) * Alphas * Alphas) # todo nn_outputs

        # The following K is respect to w(angle)
        B10 = B_F * self.Ms.unsqueeze(0) / self.I
        B10 = B10.sum(dim=1)
        B11 = torch.ones(batch_size, dtype=torch.float, device=self.device) / self.I
        B1 = torch.stack([B10, B11], dim=1)

        B = torch.stack([B0, B1], dim=1)


        #########################
        #   U (1,1,Tx,Ty)       #
        #########################
        U0 = torch.zeros(batch_size, 2, dtype=torch.float, device=self.device)
        U0[:, 0] = 1
        U1 = U0 # todo

        if not self.speed_mode:
            #############################
            #   Accurate Simulation     #
            #############################
            M = torch.hstack([torch.dstack([A*self.dt, B*self.dt, torch.zeros((batch_size, 2, 2), dtype=torch.float, device=self.device)]),
                              torch.dstack([torch.zeros((batch_size, 2, 4), dtype=torch.float, device=self.device),  torch.eye(2).unsqueeze(0).repeat(batch_size, 1, 1)]),
                              torch.zeros((batch_size, 2, 6), dtype=torch.float, device=self.device)])

            expMT = torch.matrix_exp(M)
            Ad = expMT[:, :2, :2]
            Bd1 = expMT[:, :2, 4:]
            Bd0 = expMT[:, :2, 2:4] - Bd1

            SSout = (torch.bmm(Ad, SS.unsqueeze(2)) + torch.bmm(Bd0, U0.unsqueeze(2)) + torch.bmm(Bd1, U1.unsqueeze(2)))
        else:
            #############################
            #   Simplified Simulation   #
            #############################
            # The simplified simulation addition instead of integration
            # xdot = A x + B u
            SSout = (torch.bmm(A*self.dt, SS.unsqueeze(2)) + torch.bmm(B*self.dt, U0.unsqueeze(2)) + SS.unsqueeze(2)).squeeze(2)

        return SSout[:, 0:1], SSout.view(batch_size, 2)

    def disable_NN(self):
        # Disable the contribution of the neural network
        self.NN_ratio = 0
        
    def enable_NN(self):
        self.NN_ratio = self.designed_NN_ratio

    def set_NN(self, NN_ratio):
        self.NN_ratio = NN_ratio

    def print_params(self):
        print(f'K0s: {self.K0s}')
        print(f'K1s: {self.K1s}')
        print(f'L0s: {self.L0s}')
        print(f'L1s: {self.L1s}')
        print(f'Moment arms: {self.Ms}')
        print(f'I: {self.I}')
        print(f'B: {self.B}')
        print(f'K: {self.K}')

class compensational_nn(nn.Module):
    """ This class is the fully connected neural network to compensate the stiffness generated from the
        bilinear model.
    """

    def __init__(self, device):
        """
        Args:
            ranges (list): A list of numbers which are the ranges of each output
        """
        # Generate fully connected neural network.
        super(compensational_nn, self).__init__()
        self.device = device

        # The inputs are [muscle_activation, muscle_length, muscle_speed]
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        # The neural network takes the activation of the muscle and the muscle's length and length change rate as input.
        # It outputs the stiffnesse length compenstate for the muscle.
        # The output is the force from the muscle [F]

    def forward(self, L, dL_dt, a):
        """ Calculate stiffness from the current state and the input activation of the muscle.
            The inputs are [muscle_activation, muscle_length, muscle_speed]
            The outputs are [K_nn, L_nn]. Note that, L here is not the muscle length. It is the correction of L1 in the bilinear model. L1_new = L1 + L_nn
            The outputs can only range from -1 to 1
        Args:
            L: Current muscle length
            dL_dt: Current muscle speed
            a: Muscle activation
        """

        input_tensor = torch.hstack([L, dL_dt, a]).to(self.device)
        output = self.net(input_tensor)

        # Hard clamp
        # output = torch.clamp(x, -1, 1)
        
        # The output is [F]
        return output
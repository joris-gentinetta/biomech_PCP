# BUGMAN Jan 29 2022
"""
This class simulates the joint which contains:
Several pieces of muscles
Rotational inertias for each axis, their reference values are: (values come from the opensim model2392)
I_up_down,  kg*m2   0.008
I_in_out,   kg*m2   0.004
"""

import time
import torch
from torch import nn
from torch.functional import F
import numpy as np
from Dynamics2.Muscle import Muscle
from Dynamics2.Joint import Joint
# from Muscle import Muscle
# from Joint import Joint


class Joint_2dof(Joint):
    def __init__(self, device, muscles, inertias, Lr_scale, NN_ratio, \
                 K0_scale=2000, K1_scale=40000, L0_scale=0.03, L1_scale=0.006, \
                 I_scale=0.008, M_scale=0.05, speed_mode=False):
        super().__init__(device, muscles, inertias, Lr_scale, \
                         K0_scale, K1_scale, L0_scale, L1_scale, I_scale, M_scale)
        # For the __compensational_nn
        self.compensational_nns = nn.ModuleList(
            [compensational_nn(device=self.device) for i in range(self.muscle_num)])
        self.speed_mode = speed_mode
        self.NN_ratio = NN_ratio

    def forward(self, SS, Alphas, dt=0.0166667):
        """Calculate the Joint dynamic for one step
        Output the new system state

        Args:
            SS (torch.tensor): System states. They are [wx, wy, dwx, dwy]. 
            Alphas (torch.tensor): Muscle activations, [batch_size * muscle_number]
            dt (float, optional): Delta t between each iteration. Defaults to 0.0166667.
        """
        batch_size = len(Alphas)
        # print("batch size:", batch_size)
        
        
        # Scale parameters back
        K0s = [torch.abs(self.K0s[i]*self.K0_scale*self.Lr_scale)
               for i in range(self.muscle_num)]
        K1s = [torch.abs(self.K1s[i]*self.K1_scale*self.Lr_scale)
               for i in range(self.muscle_num)]
        L0s = [torch.abs(self.L0s[i]*self.L0_scale*self.Lr_scale)
               for i in range(self.muscle_num)]
        L1s = [torch.abs(self.L1s[i]*self.L1_scale*self.Lr_scale)
               for i in range(self.muscle_num)]
        Ms = [self.Ms[i]*self.M_scale *
              self.Lr_scale for i in range(self.muscle_num)]
        I = torch.abs(self.I*self.I_scale*self.Lr_scale)
        
        
        #################################
        # Calculate the offset          #
        #################################
        """
        Because of the property of the mechanical model used here,
        the positions in the system state will not be zero when there are no muscle activations.
        So a offset is needed here to make sure cursor to be at center.
        The offset is calculated based on the K0s, L0s and moment arms.
        """
        # BB = torch.tensor([0,0], dtype = torch.float, device=self.device)
        # AA = torch.zeros((2,2), dtype = torch.float, device=self.device)
        # for i in range(self.muscle_num):
        #     BB += K0s[i]*Ms[i]*L0s[i]
        #     AA += K0s[i]*torch.matmul(Ms[i].view(-1,1), Ms[i].view(1,-1))
        # offset = -torch.linalg.solve(AA,BB)
        # Since the offset will always
        # offset = offset.detach()
        # print(offset)

        # Alpha is the clamped muscle activation. It should between 0 to 1
        # Alpha's shape is (batch_size, muscle_number)

        Alphas = torch.clamp(Alphas, 0, 1)

        # Get muscles' states and neural networks' outputs
        muscle_SSs = []
        nn_outputs = []
        for i in range(self.muscle_num):
            # Calculate Muscle States from the joint state. Muscle States are L and dL/dt for each muscle
            # [[wx, wy]]*batch_size
            w = SS[:, 0:2].view(batch_size, 1, -1)
            # [[dwx, dwy]]*batch_size
            dw_dt = SS[:, 2:4].view(batch_size, 1, -1)
            moment_arm = self.Ms[i].view(-1, 1)
            l = torch.matmul(w, moment_arm)[:, 0]             # Muscle length
            # Muscle length changing speed
            dl_dt = torch.matmul(dw_dt, moment_arm)[:, 0]
            muscle_SSs.append((l, dl_dt))
            # Neural Network
            # Each neural network's output is in the form of [k, l]
            nn_output = self.compensational_nns[i](l, dl_dt, Alphas[:, i].view(batch_size, 1))*self.NN_ratio
            # Scale nn_output by model scale factor
            nn_output = nn_output * torch.tensor(np.array([self.K1_scale, self.L1_scale]), dtype=torch.float, device=self.device)
            # print("nn_output", nn_output)
            nn_outputs.append(nn_output)
        # print("NN_ratio", self.NN_ratio)
        # print("nn outputs:", nn_outputs)

        # Compute the dynamic model
        """
        System state simulation
        xdot = A x + B u
        # Linear interpolation between steps
        # Algorithm: to integrate from time 0 to time dt, with linear
        # interpolation between inputs u(0) = u0 and u(dt) = u1, we solve
        #   xdot = A x + B u,        x(0) = x0
        #   udot = (u1 - u0) / dt,   u(0) = u0.
        #
        # Solution is
        #   [ x(dt) ]       [ A*dt  B*dt  0 ] [  x0   ]
        #   [ u(dt) ] = exp [  0     0    I ] [  u0   ]
        #   [u1 - u0]       [  0     0    0 ] [u1 - u0]
        
        In this case,
        A = [[0, 1], [-(k1+k2)/I, -b/I]]
        B = [[0, 0], [(k2*L2-k1*L1)/I, 1/I]]
        X = [Position, Speed]
        Args:
            X : System States [Position, Speed]
            a1, a2 ([type]): The activation level of the two muscle.
            dt : Time between this frame to last frame
        """

        #################
        #   Matrix A    #
        #################
        # For matrix A: A00, A01, A10, A11 are all 2x2 matrix
        # System states are [Wx, Wy, dWx_dt, dWy_dt]
        # A00 = torch.tensor(np.array([[0,0],[0,0]]*batch_size), dtype=torch.float, device=self.device)
        A00 = torch.zeros(batch_size, 2, 2,
                          dtype=torch.float, device=self.device)
        A01 = torch.tensor(
            np.array([np.eye(2)]*batch_size), dtype=torch.float, device=self.device)
        A0 = torch.dstack([A00, A01])
        K_00 = 0
        K_01 = 0
        K_10 = 0
        K_11 = 0

        # print("nn_outputs[0][:,0]", nn_outputs[0][:,0].view(batch_size, 1))
        # print("K1s[0]", K1s[0])
        for i in range(self.muscle_num):
            K_muscle = K0s[i] + (K1s[i] + nn_outputs[i][:,0].view(batch_size, 1))*Alphas[:, i].view(batch_size, 1)
            # The following K is respect to w(angle)
            K_00 += K_muscle*Ms[i][0]*Ms[i][0]
            K_01 += K_muscle*Ms[i][0]*Ms[i][1]
            K_10 += K_muscle*Ms[i][1]*Ms[i][0]
            K_11 += K_muscle*Ms[i][1]*Ms[i][1]

        A10_0 = torch.hstack([-K_00/I[0], -K_01/I[0]])
        A10_1 = torch.hstack([-K_10/I[1], -K_11/I[1]])
        A10 = torch.stack([A10_0, A10_1], 1)

        #############################
        #   DAMPING                 #
        #############################
        # Calculate DAMPING
        # TODO: #2 To make critical damping
        # Currently use unaccurated damping for place holder
        D_x = torch.sqrt(K_00*I[0])*2
        D_y = torch.sqrt(K_11*I[1])*2
        # D_x = torch.zeros(batch_size,1)
        # D_y = torch.zeros(batch_size,1)
        A11 = torch.stack([torch.hstack([-D_x/I[0], torch.zeros(batch_size, 1, device=self.device)]),
                           torch.hstack([torch.zeros(batch_size, 1, device=self.device), -D_y/I[1]])], 1)
        A1 = torch.dstack([A10, A11])
        A = torch.hstack([A0, A1])
        # print("A shape:",A.shape)
        # print("A:",A)

        #########################
        #   MATRIX B            #
        #########################
        B0 = torch.zeros(batch_size, 2, 4, dtype=torch.float,
                         device=self.device)
        B10_00 = 0
        B10_01 = 0
        B10_10 = 0
        B10_11 = 0
        for i in range(self.muscle_num):
            B_F = (K0s[i] + (K1s[i] + nn_outputs[i][:, 0].view(batch_size, 1))*Alphas[:, i].view(batch_size, 1)) * \
                (L0s[i] + (L1s[i] + nn_outputs[i][:, 1].view(batch_size, 1))
                 * Alphas[:, i].view(batch_size, 1))
            # The following K is respect to w(angle)
            B10_00 += B_F*Ms[i][0]
            B10_11 += B_F*Ms[i][1]
        B10_0 = torch.hstack(
            [B10_00/I[0], torch.zeros(batch_size, 1, device=self.device)])
        B10_1 = torch.hstack(
            [torch.zeros(batch_size, 1, device=self.device), B10_11/I[1]])
        B10 = torch.stack([B10_0, B10_1], 1)
        B11 = torch.tensor([[[1/I[0], 0], [0, 1/I[1]]]] *
                           batch_size, dtype=torch.float, device=self.device)
        B1 = torch.dstack([B10, B11])
        B = torch.hstack([B0, B1])
        # print("B:", B)

        #########################
        #   U (1,1,Tx,Ty)       #
        #########################
        U0 = torch.tensor(
            np.array([[[1, 1, 0, 0]]]*batch_size), dtype=torch.float, device=self.device)
        U1 = torch.tensor(
            np.array([[[1, 1, 0, 0]]]*batch_size), dtype=torch.float, device=self.device)

        if(self.speed_mode == False):
            #############################
            #   Accurate Simulation     #
            #############################
            M = torch.hstack([torch.dstack([A*dt, B*dt, torch.zeros((batch_size, 4, 4), dtype=torch.float, device=self.device)]),
                              torch.dstack([torch.zeros((batch_size, 4, 8), dtype=torch.float, device=self.device), torch.tensor(
                                  np.array([np.eye(4)]*batch_size), dtype=torch.float, device=self.device)]),
                              torch.zeros((batch_size, 4, 12), dtype=torch.float, device=self.device)])
            # print("M:", M)

            expMT = torch.matrix_exp(M)
            Ad = expMT[:, :4, :4]
            Bd1 = expMT[:, :4, 8:]
            Bd0 = expMT[:, :4, 4:8] - Bd1
            SSout = (torch.bmm(Ad, SS.view(batch_size, 4, 1)) + torch.bmm(Bd0,
                     U0.view(batch_size, 4, 1)) + torch.bmm(Bd1, U1.view(batch_size, 4, 1)))
        else:
            #############################
            #   Simplified Simulation   #
            #############################
            # The simplified simulation addition instead of intergration
            # xdot = A x + B u
            SSout = torch.bmm(A*dt, SS.view(batch_size, 4, 1)) + torch.bmm(B*dt, U0.view(batch_size, 4, 1)) + SS.view(batch_size, 4, 1)
        
        # return SSout.view(batch_size, 4)[:,0:2] + offset , SSout.view(batch_size, 4)
        return SSout.view(batch_size, 4)[:,0:2] , SSout.view(batch_size, 4)

    def print_params(self):
        
        K0s = [torch.abs(self.K0s[i]*self.K0_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        K1s = [torch.abs(self.K1s[i]*self.K1_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        L0s = [torch.abs(self.L0s[i]*self.L0_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        L1s = [torch.abs(self.L1s[i]*self.L1_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        Ms = [(self.Ms[i]*self.M_scale *self.Lr_scale).detach().cpu().numpy()
              for i in range(self.muscle_num)]
        I = torch.abs(self.I*self.I_scale*self.Lr_scale).detach().cpu().numpy()
        
        print("K0s:", K0s)
        print("K1s:", K1s)
        print("L0s:", L0s)
        print("L1s:", L1s)
        print("Moment arms:", Ms)
        print("I:", I)
        

class compensational_nn(nn.Module):
    """ This class is the fully connected neural network to compensate the stiffness generated from the
        bilinary model.
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
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 2)
        # The neural network takes the activation of the muscle and the muscle's length and length change rate as input.
        # It outputs the stiffnesse length compenstate for the muscle.
        # The outputs are [k, l]

        self.dropout1 = nn.Dropout(0.5)

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
        batch_size = len(a)
        # Print input
        # print("L=", L, "dL=", dL_dt, "a=", a)
        input_tensor = torch.hstack([L, dL_dt, a])
        # print("NN_input", input_tensor)
        x = self.fc1(input_tensor)
        x = F.leaky_relu(x)
        # x = self.dropout1(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        # x = self.dropout1(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        # x = self.dropout1(x)
        x = self.fc4(x)
        
        # Soft clamp
        output = torch.tanh(x)
        
        # Hard clamp
        # output = torch.clamp(x, -1, 1)
        
        # The outputs are [k, l]
        print(output)
        return output


# Test
"""
device = torch.device("cpu")
m1 = Muscle(2000, 40000, 0.03, 0.006, [0.05, 0])
m2 = Muscle(2000, 40000, 0.03, 0.006, [-0.05, 0])
m3 = Muscle(2000, 40000, 0.03, 0.006, [0, 0.05])
m4 = Muscle(2000, 40000, 0.03, 0.006, [0, -0.05])
model = Joint_2dof(device, [m1, m2, m3, m4], [
                   0.004, 0.008], Lr_scale=5, NN_ratio=0.0, speed_mode=False)
# model = Joint_2dof(device, [m1, m2, m3, m4], [0.004, 0.008], Lr_scale=5, NN_ratio=0.0, speed_mode=True)
model.eval()
X = torch.FloatTensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
a1 = torch.FloatTensor([[0, 0, 0, 0], [0.1, 0, 1, 0], [0, 0, 1, 0]])
# For time benchmark
start = time.time()
for i in range(20):
    X = model(X, a1)
    # print(X)
    print(X[1, 0])
end = time.time()
print(end - start)

"""

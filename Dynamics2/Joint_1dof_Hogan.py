# Mikey Fernandez 07/19/2023
"""
The joint with only 1 degree of freedom
In this model, the total torque is the sum of Hogan antagonist model torque, with compensation stiffness and compensation isometric torque provided with neural networks
"""

import torch
from torch import nn
from Dynamics2.Joint_Hogan import Joint

class Joint_1dof(Joint):
    def __init__(self, device, muscles, parameters, Lr_scale, NN_ratio):
        super().__init__(device, muscles, parameters['I'], parameters['B'], Lr_scale, parameters['I_'], parameters['B_'], parameters['K_'], parameters['T_'])
        # (self, device, muscles, inertias, damping, Lr_scale, I_scale=0.008, B_scale=4, K_scale=40, T_scale=100)
        self.compensational_nns = nn.ModuleList([compensational_nn(device=self.device) for _ in range(self.muscle_num)])
        self.designed_NN_ratio = NN_ratio
        self.NN_ratio = NN_ratio

        self.order = parameters['order']
        self.setCoefficients()

    def forward(self, SS, Alphas, dt=0.0166667):
        """Calculate the Joint dynamic for one step
        Output the new system state

        Args:
            SS (torch.tensor): System states. They are [theta, omega]. 
            Alphas (torch.tensor): Muscle activations, [batch_size * muscle_number]
            dt (float, optional): Delta t between each iteration. Defaults to 0.0166667.

        System state simulation

        theta_dot = omega
        I*omega_dot = sum_(i=0)^(muscle_num) (T[i]*(1 + T_nn[i]) + sign*th*K[i]*(1 + K_nn[i]))*alpha[i] - B*dth_dt
        """

        batch_size = len(Alphas)

        # Scale parameters back
        Ts = [torch.abs(self.Ts[i]*self.T_scale*self.Lr_scale) for i in range(self.muscle_num)]
        Ks = [torch.abs(self.Ks[i]*self.K_scale*self.Lr_scale) for i in range(self.muscle_num)]
        
        I = torch.abs(self.I*self.I_scale*self.Lr_scale)
        B_joint = torch.abs(self.B*self.B_scale*self.Lr_scale)
        
        # Alpha is the clamped muscle activation. It should between 0 to 1. Alpha's shape is (batch_size, muscle_number)
        Alphas = torch.clamp(Alphas, 0, 1)

        k = [torch.zeros_like(SS) for _ in range(self.order)]

        runSum = []
        
        #### need to do this first one separately
        # Extract the joint state
        th = SS[:, 0].view(batch_size, -1) # [[th]]*batch_size
        dth_dt = SS[:, 1].view(batch_size, -1) # [[dth_dt]]*batch_size

        # Get muscles' states and neural networks' outputs
        muscleTor = [] # torch.zeros_like(th)
        for i in range(self.muscle_num):
            sign = 1 if not i else -1 # negate the position/velocity so the NN operates the same way on both muscles (in terms of lengthening/shortening)
            
            act = Alphas[:, i].view(batch_size, -1)

            # Each neural network's output is in the form of [T_nn, K_nn], which are each numbers ranging between -1 and 1
            nn_output = self.compensational_nns[i](sign*th, sign*dth_dt, act)*self.NN_ratio
            T_nn, K_nn = (nn_output[:, 0].view(batch_size, -1), nn_output[:, 1].view(batch_size, -1)) # extract these

            T_mus = -sign*(Ts[i]*(1 + T_nn) + sign*Ks[i]*th.mul(1 + K_nn)).mul(act) # this is the entire muscle's force equation!

            muscleTor.append(T_mus)

        omega_dot = (sum(muscleTor) - B_joint*dth_dt)/I

        k[0] = dt*torch.hstack([dth_dt, omega_dot])
        assert(k[0].shape == SS.shape)
        runSum.append(self.b[0]*k[0])
        ####

        for run in range(1, self.order):
            # tempSS = SS + torch.sum(torch.stack([a * k[j] for j, a in enumerate(self.a[run])]), dim=1)
            tempSS = SS + sum([a * k[j] for j, a in enumerate(self.a[run])])
            # Extract the joint state
            th = tempSS[:, 0].view(batch_size, -1) # [[th]]*batch_size
            dth_dt = tempSS[:, 1].view(batch_size, -1) # [[dth_dt]]*batch_size

            # Get muscles' states and neural networks' outputs
            muscleTor = [] # torch.zeros_like(th)
            for i in range(self.muscle_num):
                sign = 1 if not i else -1 # negate the position/velocity so the NN operates the same way on both muscles (in terms of lengthening/shortening)
                
                act = Alphas[:, i].view(batch_size, -1)

                # Each neural network's output is in the form of [T_nn, K_nn], which are each numbers ranging between -1 and 1
                nn_output = self.compensational_nns[i](sign*th, sign*dth_dt, act)*self.NN_ratio
                T_nn, K_nn = (nn_output[:, 0].view(batch_size, -1), nn_output[:, 1].view(batch_size, -1)) # extract these

                T_mus = -sign*(Ts[i]*(1 + T_nn) + sign*Ks[i]*th.mul(1 + K_nn)).mul(act) # this is the entire muscle's force equation!

                muscleTor.append(T_mus)

            omega_dot = (sum(muscleTor) - B_joint*dth_dt)/I

            k[run] = dt*torch.hstack([dth_dt, omega_dot])
            runSum.append(self.b[run]*k[run])

        SSout = SS + sum(runSum)

        return SSout[:, 0].view(batch_size, -1), SSout.view(batch_size, -1)
    
    def setCoefficients(self):
        """
        Set the coefficients for the specified order of the Runge-Kutta method.
        """
        if self.order == 1:
            self.a = []
            self.b = [1.0]
        elif self.order == 2:
            self.a = [[], [0.5]]
            self.b = [0.0, 1.0]
        elif self.order == 3:
            self.a = [[], [3/4], [1/3, 1/3]]
            self.b = [2/9, 3/9, 4/9]
        elif self.order == 4:
            self.a = [[], [0.5], [0.0, 0.5], [0.0, 0.0, 1.0]]
            self.b = [1/6, 1/3, 1/3, 1/6]
        else:
            raise ValueError(f"Runge-Kutta order {self.order} is not supported.")


    # def forward(self, SS, Alphas, dt=0.0166667):
    #     """Calculate the Joint dynamic for one step
    #     Output the new system state

    #     Args:
    #         SS (torch.tensor): System states. They are [theta, omega]. 
    #         Alphas (torch.tensor): Muscle activations, [batch_size * muscle_number]
    #         dt (float, optional): Delta t between each iteration. Defaults to 0.0166667.
    #     """
    #     batch_size = len(Alphas)

    #     SSout = self.integrator(SS, Alphas, dt)

    #     return SSout[:, 0].view(batch_size, -1), SSout.view(batch_size, -1)
    
    # def forwardStep(self, SS, Alphas):
    #     batch_size = len(Alphas)
        
    #     # Scale parameters back
    #     Ts = [torch.abs(self.Ts[i]*self.T_scale*self.Lr_scale) for i in range(self.muscle_num)]
    #     Ks = [torch.abs(self.Ks[i]*self.K_scale*self.Lr_scale) for i in range(self.muscle_num)]
        
    #     I = torch.abs(self.I*self.I_scale*self.Lr_scale)
    #     B_joint = torch.abs(self.B*self.B_scale*self.Lr_scale)
        
    #     # Alpha is the clamped muscle activation. It should between 0 to 1
    #     # Alpha's shape is (batch_size, muscle_number)
    #     Alphas = torch.clamp(Alphas, 0, 1)

    #     # Compute the dynamic model - note that this is a nonlinear system!
    #     """
    #     System state simulation

    #     theta_dot = omega
    #     I*omega_dot = sum_(i=0)^(muscle_num) (T[i]*(1 + T_nn[i]) + sign*th*K[i]*(1 + K_nn[i]))*alpha[i] - B*dth_dt
    #     """

    #     # Extract the joint state
    #     th = SS[:, 0].view(batch_size, -1) # [[th]]*batch_size
    #     dth_dt = SS[:, 1].view(batch_size, -1) # [[dth_dt]]*batch_size

    #     # Get muscles' states and neural networks' outputs
    #     muscleTor = torch.zeros_like(th)
    #     for i in range(self.muscle_num):
    #         sign = 1 if not i else -1 # negate the position/velocity so the NN operates the same way on both muscles (in terms of lengthening/shortening)
            
    #         act = Alphas[:, i].view(batch_size, -1)

    #         # Each neural network's output is in the form of [T_nn, K_nn], which are each numbers ranging between -1 and 1
    #         nn_output = self.compensational_nns[i](sign*th, sign*dth_dt, act)*self.NN_ratio
    #         T_nn, K_nn = (nn_output[:, 0].view(batch_size, -1), nn_output[:, 1].view(batch_size, -1)) # extract these

    #         T_mus = -sign*(Ts[i]*(1 + T_nn) + sign*Ks[i]*th.mul(1 + K_nn)).mul(act) # this is the entire muscle's force equation!

    #         muscleTor += T_mus

    #     netTor = (muscleTor - B_joint*dth_dt)/I

    #     dSSout_dt = torch.hstack([dth_dt, netTor])

    #     return dSSout_dt

    def disable_NN(self):
        # Disable the contribution of the neural network
        self.NN_ratio = 0
        
    def enable_NN(self):
        self.NN_ratio = self.designed_NN_ratio

    def set_NN(self, NN_ratio):
        self.NN_ratio = NN_ratio

    def print_params(self):
        Ts = [torch.abs(self.Ts[i]*self.T_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        Ks = [torch.abs(self.Ks[i]*self.K_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        I = torch.abs(self.I*self.I_scale*self.Lr_scale).detach().cpu().numpy()
        B = torch.abs(self.B*self.B_scale*self.Lr_scale).detach().cpu().numpy()

        print(f'Ts: {Ts}')
        print(f'Ks: {Ks}')
        print(f'I: {I}')
        print(f'B: {B}')    

class compensational_nn(nn.Module):
    """ This class is the fully connected neural network to provide position and velocity dependence of muscle stiffness/isometric torque """

    def __init__(self, device):
        """
        Args:
            device (str): the device where the model is run
        """
        # Generate fully connected neural network.
        super(compensational_nn, self).__init__()
        self.device = device

        # The inputs are [joint position, joint velocity, muscle activation]
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 32),
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2), 
            nn.Linear(32, 32),
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 2),
            nn.Tanh()
        )

    def forward(self, th, dth_dt, a):
        """ Calculate torque, stiffness from the current state of the joint and the activation of the muscle.
            The inputs are [joint position, joint velocity, muscle activation]
            The outputs are [T_nn, K_nn]
            The outputs can only range from -1 to 1
        Args:
            L: Current muscle length
            dL_dt: Current muscle speed
            a: Muscle activation
        """

        input_tensor = torch.hstack([th, dth_dt, a]).to(self.device)
        output = self.net(input_tensor)

        return output
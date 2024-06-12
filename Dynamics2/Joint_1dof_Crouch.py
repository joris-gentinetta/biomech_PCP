# Mikey Fernandez 07/19/2022

"""
AMI 1DoF joint based on the Crouch 2016 muscle model
"""

import torch
import numpy as np
from Dynamics2.Joint_Crouch import Joint
import time

class Joint_1dof(Joint):
    def __init__(self, device, muscles, parameters, Lr_scale, NN_ratio):
        super().__init__(device, muscles, parameters['I'], parameters['K'], parameters['B'], parameters['L'], Lr_scale, \
                         parameters['K_'], parameters['B_'], parameters['I_'], parameters['fMax_'], parameters['lOpt_'], parameters['kPE_'], parameters['vMax_'], parameters['W_'], parameters['c_'])
 
        self.designed_NN_ratio = NN_ratio
        self.NN_ratio = NN_ratio

    def RungeKutta(self, dt, theta, omega, Alphas, f_maxs, lM_opts, kPEs, v_maxs, Ws, cs):
        k1s = self.deriv(theta, omega, Alphas, f_maxs, lM_opts, kPEs, v_maxs, Ws, cs)
        k2s = self.deriv(theta + k1s[0]*dt/2, omega + k1s[1]*dt/2, Alphas, f_maxs, lM_opts, kPEs, v_maxs, Ws, cs)
        k3s = self.deriv(theta + k2s[0]*dt/2, omega + k2s[1]*dt/2, Alphas, f_maxs, lM_opts, kPEs, v_maxs, Ws, cs)
        k4s = self.deriv(theta + k3s[0]*dt, omega + k3s[1]*dt, Alphas, f_maxs, lM_opts, kPEs, v_maxs, Ws, cs)

        return [1/6*k1s[i] + 1/3*k2s[i] + 1/3*k3s[i] + 1/6*k4s[i] for i in range(2)]

    # calculate the state derivatives
    def deriv(self, theta, omega, alphas, f_maxs, lM_opts, kPEs, v_maxs, Ws, cs):
        # calculate the moment arm in here
        lMTs = []
        rMTs = []
        for i in range(self.muscle_num):
            lMTs.append(torch.sqrt(self.L2**2 + self.L3**2 + (-1)**(i + 1)*2*self.L2*self.L3*torch.cos(theta + np.pi/2)))
            rMTs.append((-1)**(i + 1)*self.L2*self.L3*torch.sin(theta + np.pi/2)/lMTs[i])
 
        # print(f'theta: {theta}')
        # print(f'omega: {omega}')

        # muscle dynamics
        fM = []
        for i in range(self.muscle_num):
            vCE = (rMTs[i]*omega)/lM_opts[i]
            extended = torch.where(lMTs[i] > lM_opts[i], lMTs[i] - lM_opts[i], torch.zeros_like(lM_opts[i]))
            fL_ce = torch.mul(f_maxs[i], 1 - torch.div(torch.square(lMTs[i] - lM_opts[i]), torch.square(torch.mul(Ws[i], lM_opts[i]))))
            fv_ce = torch.div(v_maxs[i] - vCE, v_maxs[i] + torch.div(vCE, cs[i]))
            f_pe = torch.mul(kPEs[i], torch.square(extended))
            fM.append(alphas[:, i][:, None, None]*torch.mul(fL_ce, fv_ce) + f_pe)

            # print(f'---- muscle {i} ----')
            # print(f'r: {rMTs[i]} | l: {lMTs[i]}')
            # print(f'lM_opt: {lM_opts[i]}')
            # print(f'vCe: {vCE}')
            # print(f'extended: {extended}')
            # print(f'fL_ce: {fL_ce}')
            # print(f'fv_ce: {fv_ce}')
            # print(f'f_pe: {f_pe}')
            # print(f'fM: {fM[i]}')
        
        # joint dynamics
        theta_dt = omega
        omega_dt = (-self.K*theta - self.B*omega - rMTs[0]*fM[0] - rMTs[1]*fM[1])/self.I
        
        return theta_dt, omega_dt

    def forward(self, SS, Alphas, dt=0.0166667):
        """
        Calculate the Joint dynamics for one step
        Output the new system state
 
        Args:
        SS (torch.tensor): System states. They are [theta, omega, fM0, fM1] - the joint position and velocity, the muscle forces
        Alphas (torch.tensor): Muscle activations, [batch_size * muscle_number]
        dt (float, optional): Delta t between each iteration. Defaults to 0.0166667.
        """
        batch_size = len(Alphas)
        
        # Scale parameters back
        f_maxs = [torch.abs(self.f_maxs[i]*self.f_max_scale*self.Lr_scale) for i in range(self.muscle_num)]
        lM_opts = [torch.abs(self.lM_opts[i]*self.lM_opt_scale*self.Lr_scale) for i in range(self.muscle_num)]
        kPEs = [torch.abs(self.kPEs[i]*self.kPEs_scale*self.Lr_scale) for i in range(self.muscle_num)]
        v_maxs = [torch.abs(self.vMaxs[i]*self.vMaxs_scale*self.Lr_scale) for i in range(self.muscle_num)]
        Ws = [torch.abs(self.Ws[i]*self.W_scale*self.Lr_scale) for i in range(self.muscle_num)]
        cs = [torch.abs(self.cs[i]*self.c_scale*self.Lr_scale) for i in range(self.muscle_num)]
               
        # Get muscles' states
        theta = SS[:, 0:1].view(batch_size, 1, -1) # [[wx, wy]]*batch_size
        omega = SS[:, 1:2].view(batch_size, 1, -1)
 
        # Compute the dynamic model using 4th order Runge Kutta
        m = self.RungeKutta(dt, theta, omega, Alphas, f_maxs, lM_opts, kPEs, v_maxs, Ws, cs)
 
        # Euler forward (mostly for testing, only one call per loop)
        # m = self.deriv(theta, omega, Alphas, f_maxs, lM_opts, kPEs, v_maxs, Ws, cs)
 
        SSout = torch.hstack((theta + m[0]*dt, omega + m[1]*dt))

        # print(f'm: {[m[i].detach().cpu().numpy() for i in range(self.muscle_num)]}')
        # time.sleep(1)
 
        return SSout.view(batch_size, 2)[:, 0:1], SSout.view(batch_size, 2)

    def print_params(self):
        f_maxs = [torch.abs(self.f_maxs[i]*self.f_max_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        lM_opts = [torch.abs(self.lM_opts[i]*self.lM_opt_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        kPEs = [torch.abs(self.kPEs[i]*self.kPEs_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        v_maxs = [torch.abs(self.vMaxs[i]*self.vMaxs_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        Ws = [torch.abs(self.Ws[i]*self.W_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        cs = [torch.abs(self.cs[i]*self.c_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
 
        K = torch.abs(self.K*self.K_scale*self.Lr_scale).detach().cpu().numpy()
        B = torch.abs(self.B*self.B_scale*self.Lr_scale).detach().cpu().numpy()
        I = torch.abs(self.I*self.I_scale*self.Lr_scale).detach().cpu().numpy()
        
        print(f'f_maxs: {f_maxs}')
        print(f'lM_opts: {lM_opts}')
        print(f'kPEs: {kPEs}')
        print(f'v_maxs: {v_maxs}')
        print(f'Ws: {Ws}')
        print(f'cs: {cs}')
        print(f'K: {K}')
        print(f'B: {B}')
        print(f'I: {I}')
        print()

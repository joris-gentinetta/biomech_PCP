# BUGMAN Feb 17 2022
# Modified by Mikey Fernandez 04/02/2022
"""
AMI 1DoF joint based on the Shadmehr-Arbib muscle model
"""

import torch
import numpy as np
from Dynamics2.Muscle_Shadmehr import Muscle
from Dynamics2.Joint_Shadmehr import Joint

class Joint_1dof(Joint):
#      def __init__(self, device, muscles, inertias, K, B, lengths, Lr_scale, NN_ratio, \
#                  K_scale=1, B_scale=1, I_scale=1, f_max_scale=1, lM_opt_scale=1, kSEs_scale=1, kPEs_scale=1, bs_scale=1, gamma_scale=1):
    def __init__(self, device, muscles, parameters, Lr_scale, NN_ratio):
        super().__init__(device, muscles, parameters['I'], parameters['K'], parameters['B'], parameters['L'], Lr_scale, \
                         parameters['K_'], parameters['B_'], parameters['I_'], parameters['fMax_'], parameters['lOpt_'], parameters['kSE_'], parameters['kPE_'], parameters['b_'], parameters['gamma_'])
 
        self.designed_NN_ratio = NN_ratio
        self.NN_ratio = NN_ratio

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
        Alphas = torch.clamp(Alphas, 0, 1) # this must be nonnegative
        
        # Scale parameters back
        f_maxs =  [torch.abs(self.f_maxs[i]*self.f_max_scale*self.Lr_scale) for i in range(self.muscle_num)]
        lM_opts = [torch.abs(self.lM_opts[i]*self.lM_opt_scale*self.Lr_scale) for i in range(self.muscle_num)]
        kSEs =    [torch.abs(self.kSEs[i]*self.kSEs_scale*self.Lr_scale) for i in range(self.muscle_num)]
        kPEs =    [torch.abs(self.kPEs[i]*self.kPEs_scale*self.Lr_scale) for i in range(self.muscle_num)]
        bs =      [torch.abs(self.bs[i]*self.bs_scale*self.Lr_scale) for i in range(self.muscle_num)]
        gammas =  [torch.abs(self.gammas[i]*self.gamma_scale*self.Lr_scale) for i in range(self.muscle_num)]
        B =        torch.abs(self.B*self.B_scale*self.Lr_scale)
        K =        torch.abs(self.K*self.K_scale*self.Lr_scale)
        I =        torch.abs(self.I*self.I_scale*self.Lr_scale)
 
        # Get muscles' states
        theta = SS[:, 0:1].view(batch_size, 1, -1) # [[theta]]*batch_size
        omega = SS[:, 1:2].view(batch_size, 1, -1)
        fMs   = SS[:, 2:4].view(batch_size, 2, -1) # [[fM0, fM1]]*batch_size

       #  print(f'---------\ntheta   : {theta}')
       #  print(f'omega   : {omega}')
       #  print(f'fM{0}     : {fMs[:, 0]}')
       #  print(f'fM{1}     : {fMs[:, 1]}')
       #  print(f'B: {B}')
       #  print(f'K: {K}')
       #  print(f'I: {I}')

        # calculate the moment arm in here
        lMTs = []
        rMTs = []
        dldts = []
        for i in range(self.muscle_num):
            lMT = torch.sqrt(self.L2**2 + self.L3**2 + (-1)**i*2*self.L2*self.L3*torch.cos(theta + np.pi/2)).clone().detach().requires_grad_(True)
            rMT = (-1)**(i + 1)*self.L2*self.L3*torch.sin(theta + np.pi/2)/lMT
            dldt = torch.mul(rMT, omega)
            lMTs.append(lMT)
            rMTs.append(rMT)
            dldts.append(dldt)
 
        fM_dt = []
        # muscle dynamics
        for i in range(self.muscle_num):
            lM_norm = lMTs[i]/lM_opts[i] # normalize length
            pos = -torch.square(lM_norm - torch.ones_like(lM_norm, requires_grad=True))/gammas[i]
            S = torch.exp(pos)
            mStretch = torch.maximum(lMTs[i] - lM_opts[i], torch.zeros_like(lMTs[i], requires_grad=True)) # add this requires grad to avoid errors
            mShortening = torch.minimum(dldts[i], torch.zeros_like(dldts[i], requires_grad=True))
            f_AL = torch.mul(Alphas[:, i, None, None], f_maxs[i]*S) # get active force
            onePlus = torch.ones_like(kPEs[i], requires_grad=True) + kPEs[i]/kSEs[i]
       #      f_cont = onePlus*fMs[:, i, None]
            f_cont = onePlus*torch.mul(fMs[:, i, None], mStretch)
            b_cont = bs[i]*torch.mul(dldts[i], mShortening)
            PE_cont = kPEs[i]*mStretch
            fM_dt.append((kSEs[i]/bs[i])*(PE_cont + b_cont - f_cont + f_AL))

            lMTs[i].retain_grad()
            lMTs[i].register_hook(lambda x: print(f'lMT{i}: {x}'))
            rMTs[i].retain_grad()
            rMTs[i].register_hook(lambda x: print(f'rMT{i}: {x}'))
            kPEs[i].retain_grad()
            kPEs[i].register_hook(lambda x: print(f'kPE{i}: {x}'))
            kSEs[i].retain_grad()
            kSEs[i].register_hook(lambda x: print(f'kSE{i}: {x}'))
            lM_opts[i].retain_grad()
            lM_opts[i].register_hook(lambda x: print(f'lM_opt{i}: {x}'))
            gammas[i].retain_grad()
            gammas[i].register_hook(lambda x: print(f'gamma{i}: {x}'))
            f_maxs[i].retain_grad()
            f_maxs[i].register_hook(lambda x: print(f'f_maxs{i}: {x}'))
            bs[i].retain_grad()
            bs[i].register_hook(lambda x: print(f'b{i}: {x}'))
            lM_norm.retain_grad()
            lM_norm.register_hook(lambda x: print(f'lM_norm{i}: {x}'))
            pos.retain_grad()
            pos.register_hook(lambda x: print(f'pos{i}: {x}'))
            S.retain_grad()
            S.register_hook(lambda x: print(f'S{i}: {x}'))
            mStretch.retain_grad()
            mStretch.register_hook(lambda x: print(f'mStretch{i}: {x}'))
            f_AL.retain_grad()
            f_AL.register_hook(lambda x: print(f'f_AL{i}: {x}'))
            b_cont.retain_grad()
            b_cont.register_hook(lambda x: print(f'b_cont{i}: {x}'))
            onePlus.retain_grad()
            onePlus.register_hook(lambda x: print(f'onePlus{i}: {x}'))
            f_cont.retain_grad()
            f_cont.register_hook(lambda x: print(f'f_cont{i}: {x}'))
            PE_cont.retain_grad()
            PE_cont.register_hook(lambda x: print(f'PE_cont{i}: {x}'))
            fM_dt[i].retain_grad()
            fM_dt[i].register_hook(lambda x: print(f'fM_dt{i}: {x}'))

       #      print(f'lMT{i}   : {lMTs[i]}, {lMTs[i].shape}')
       #      print(f'rM{i}    : {rMTs[i]}, {rMTs[i].shape}')
       #      print(f'dl/dt{i} : {torch.mul(rMTs[i], omega)}, {torch.mul(rMTs[i], omega).shape}')
       #      print(f'lM_norm{i}: {lM_norm}, {lM_norm.shape}')
       #      print(f'pos{i}   : {pos}, {pos.shape}')
       #      print(f'S{i}     : {S}. {S.shape}')
       #      print(f'alpha{i}: {Alphas[:, i]}, {Alphas[:, i].shape}')
       #      print(f'f_AL{i}  : {f_AL}, {f_AL.shape}')
       #      print(f'onePlus{i}: {onePlus}, {onePlus.shape}')
       #      print(f'f_cont{i}: {f_cont}, {f_cont.shape}')
       #      print(f'b_cont{i}: {b_cont}, {b_cont.shape}')
       #      print(f'PE_cont{i}: {PE_cont}, {PE_cont.shape}')
       #      print(f'fM_dt{i}  : {fM_dt[i]}, {fM_dt[i].shape}')
       #      print(f'f_maxs{i}: {f_maxs[i]}, {f_maxs[i].shape}')
       #      print(f'lM_opts{i}: {lM_opts[i]}, {lM_opts[i].shape}')
       #      print(f'kSEs{i}: {kSEs[i]}, {kSEs[i].shape}')
       #      print(f'kPEs{i}: {kPEs[i]}, {kPEs[i].shape}')
       #      print(f'bs{i}: {bs[i]}, {bs[i].shape}')
       #      print(f'gammas{i}: {gammas[i]}, {gammas[i].shape}')
 
         # joint dynamics
        muscleTorque = torch.mul(rMTs[0], fMs[:, 0, None]) + torch.mul(rMTs[1], fMs[:, 1, None])

        theta_dt = omega
        omega_dt = (-K*theta - B*omega - muscleTorque)/I

        m = torch.hstack((theta_dt, omega_dt, fM_dt[0], fM_dt[1]))
        states = torch.hstack((theta, omega, fMs[:, 0, None], fMs[:, 1, None]))
        SSout = states + m*dt

       #  print(f'torques:\n\tspring: {-K*theta} ({(-K*theta).shape})\t|\tdamp: {-B*omega} ({(-B*omega).shape})\t|\tmTorque: {-muscleTorque} ({muscleTorque.shape})')
       #  print(f'm:\n{m}')
       #  print(f'SSout:\n{SSout}')

        #### Delete me
       #  m.retain_grad()
       #  m.register_hook(lambda x: print(f'm{i}: {x}'))
       #  SSout.retain_grad()
       #  SSout.register_hook(lambda x: print(f'SSout{i}: {x}'))
       #  muscleTorque.retain_grad()
       #  muscleTorque.register_hook(lambda x: print(f'muscleTorque{i}: {x}'))
 
        return SSout.view(batch_size, 4)[:, 0:1], SSout.view(batch_size, 4)

        ### Code that may be readded
    #     drM0 = self.L2*self.L3*torch.div(torch.mul(lMT0, torch.cos(theta + np.pi/2)) - torch.mul(torch.sin(theta + np.pi/2), rM0), torch.square(lMT0))
    #     drM1 = -self.L2*self.L3*torch.div(torch.mul(lMT1, torch.cos(theta + np.pi/2)) - torch.mul(torch.sin(theta + np.pi/2), rM0), torch.square(lMT1))
    #     # lumped muscle stiffness
    #     netK0 = torch.div(kPEs[0], 1 + torch.div(kPEs[0], kSEs[0]))
    #     netK1 = torch.div(kPEs[1], 1 + torch.div(kPEs[1], kSEs[1]))
 
    #     dS0 = -2*torch.mul(S0, torch.div(lM_norm0 - 1, torch.mul(gammas[0], lM_opts[0])))
    #     dS1 = -2*torch.mul(S1, torch.div(lM_norm1 - 1, torch.mul(gammas[1], lM_opts[1])))
 
    #    #  K0 = torch.mul(dS0, torch.div(fM0 - torch.mul(netK0, lMT0 - lM_opts[0]), S0)) + netK0
    #    #  K1 = torch.mul(dS1, torch.div(fM1 - torch.mul(netK1, lMT1 - lM_opts[1]), S1)) + netK1
    #     K0 = torch.mul(dS0, torch.div(fM0 - torch.mul(netK0, mStretch0), S0)) + netK0
    #     K1 = torch.mul(dS1, torch.div(fM1 - torch.mul(netK1, mStretch1), S1)) + netK1
 
    #     Kmus = -torch.mul(drM0, fM0) - torch.mul(drM1, fM1) - torch.mul(torch.square(rM0), K0) - torch.mul(torch.square(rM1), K1) # this should be a negative number

    def print_params(self):
        f_maxs = [torch.abs(self.f_maxs[i]*self.f_max_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        lM_opts = [torch.abs(self.lM_opts[i]*self.lM_opt_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        kSEs = [torch.abs(self.kSEs[i]*self.kSEs_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        kPEs = [torch.abs(self.kPEs[i]*self.kPEs_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        bs = [torch.abs(self.bs[i]*self.bs_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
        gammas = [torch.abs(self.gammas[i]*self.gamma_scale*self.Lr_scale).detach().cpu().numpy()[0]
               for i in range(self.muscle_num)]
 
        K = torch.abs(self.K*self.K_scale*self.Lr_scale).detach().cpu().numpy()
        B = torch.abs(self.B*self.B_scale*self.Lr_scale).detach().cpu().numpy()
        I = torch.abs(self.I*self.I_scale*self.Lr_scale).detach().cpu().numpy()
        
        print(f'f_maxs: {f_maxs}')
        print(f'lM_opts: {lM_opts}')
        print(f'kSEs: {kSEs}')
        print(f'kPEs: {kPEs}')
        print(f'bs: {bs}')
        print(f'gammas: {gammas}')
        print(f'K: {K}')
        print(f'B: {B}')
        print(f'I: {I}')
        print()

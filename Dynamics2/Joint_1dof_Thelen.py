# BUGMAN Feb 17 2022
# Modified by Mikey Fernandez 04/02/2022
"""
AMI 1DoF joint based on the Thelen muscle model
"""

# from msvcrt import kbhit
import torch
from Dynamics2.Joint_Thelen import Joint
import numpy as np
import pandas as pd
np.set_printoptions(linewidth=200, precision=8, suppress=True, sign=' ', floatmode='fixed')
pd.set_option('display.width', 200)

class Joint_1dof(Joint):
    # def __init__(self, device, muscles, inertias, damping, stiffness, Lr_scale, \
    #             f_max_scale=1, lM_opt_scale=1, vM_max_scale=1, lT_slack_scale=1, pen_scale=1, moment_arm_scale=1, I_scale=1, K_scale=1, B_scale=1):
    # def __init__(self, device, muscles, inertias, damping, stiffness, Lr_scale, \
    #             f_max_=1, lM_opt_=1, vM_max_=1, lT_slack_=1, pen_=1, moment_arm_=1, I_=1, K_=1, B_=1, lIn_=1, lOrig_=1):
    def __init__(self, device, muscles, p, Lr_scale, NN_ratio):
       super().__init__(device, muscles, inertias=p['I'], damping=p['B'], stiffness=p['K'], Lr_scale=Lr_scale, \
                       f_max_=p['fMax_'], lM_opt_=p['lOpt_'], vM_max_=p['vM_'], vAlpha_=p['vAlpha_'], lT_slack_=p['lT_'], pen_=p['pen_'], lIn_=p['lIn_'], lOrig_=p['lOrig_'], I_=p['I_'], K_=p['K_'], B_=p['B_'])

       self.designed_NN_ratio = NN_ratio
       self.NN_ratio = NN_ratio

    def forward(self, SS, Alphas, dt=0.0166667, jointID=None):
       """
       Calculate the Joint dynamics for one step
       Output the new system state
       Args:
           SS (torch.tensor): System states. They are [theta, omega, lM0, lM1] - the joint position and velocity, the muscle lengths
           Alphas (torch.tensor): Muscle activations, [batch_size * muscle_number]
           dt (float, optional): Delta t between each iteration. Defaults to 0.0166667.
       """
       batch_size = len(Alphas)
       
       # Scale p back
       f_maxs = [(self.f_maxs[i]*self.f_max_scale*self.Lr_scale).abs() for i in range(self.muscle_num)]
       lM_opts = [(self.lM_opts[i]*self.lM_opt_scale*self.Lr_scale).abs() for i in range(self.muscle_num)]
       vM_maxs = [(self.vM_maxs[i]*self.vM_max_scale*self.Lr_scale).abs() for i in range(self.muscle_num)]
       vAlphas = [(self.vAlphas[i]*self.vAlpha_scale*self.Lr_scale).abs() for i in range(self.muscle_num)]
       lT_slacks = [(self.lT_slacks[i]*self.lT_slack_scale*self.Lr_scale).abs() for i in range(self.muscle_num)]
       pen_opts = [(self.pen_opts[i]*self.pen_scale*self.Lr_scale).abs() for i in range(self.muscle_num)]
       # moment_arms = [self.moment_arms[i]*self.moment_arm_scale*self.Lr_scale for i in range(self.muscle_num)]
       lIns = [self.lIns[i]*self.lIn_scale*self.Lr_scale for i in range(self.muscle_num)]
       lOrigs = [self.lOrigs[i]*self.lOrig_scale*self.Lr_scale for i in range(self.muscle_num)]
       I = (self.I*self.I_scale*self.Lr_scale).abs()
       B = (self.B*self.B_scale*self.Lr_scale).abs()
       K = (self.K*self.K_scale*self.Lr_scale).abs()

       Alphas = torch.clamp(Alphas, self.minActivation, 1) # level shift to keep a small activation always

       # Get muscles' states
       theta = SS[:, 0].view(batch_size, 1, -1) # joint angle, rad
       omega = SS[:, 1].view(batch_size, 1, -1) # joint velocity, rad/s
       lMs = [SS[:, 2].view(batch_size, 1, -1), SS[:, 3].view(batch_size, 1, -1)] # muscle lengths (internal DoF), m

       lMTs = []
       rMs = []
       vMs = []
       fMs = []
       lM_outs = []
       vals = {'lM': [], 'lMNorm': [], 'lMT': [], 'rMT': [], 'width': [], 'pen': [], 'alpha': [], 'lT': [], 'epsT': [], 'tendon': [], 'active': [], 'passive': [], 'contractile': [], 'dlM_dt': [], 'vM': []}

       ins = {'theta': theta.squeeze().detach().cpu().numpy(), 'omega': omega.squeeze().detach().cpu().numpy(), 'lM0': lMs[0].squeeze().detach().cpu().numpy(), 'lM1': lMs[1].squeeze().detach().cpu().numpy()}
       print('Input:')
       print(pd.DataFrame.from_dict(ins, orient='index'))
       # print(f'Input:\n\ttheta:\n\t\t{theta.squeeze().detach().cpu().numpy()}\n\tomega:\n\t\t{omega.squeeze().detach().cpu().numpy()}\n\tlM0:\n\t\t{lMs[0].squeeze().detach().cpu().numpy()}\n\tlM1:\n\t\t{lMs[1].squeeze().detach().cpu().numpy()}')
       
       for i in range(self.muscle_num):
           musc = self.init_muscles[i]

           # MTU state
           lMT = musc.lMT(theta, lIns[i], lOrigs[i])
           moment_arm = musc.rMT(theta, lIns[i], lOrigs[i])
           # moment_arm = moment_arms[i].view(-1, 1) # moment arm, m
       #     width = lM_opts[i]*pen_opts[i].sin() # width of muscle, m
           width = musc.width(lM_opts[i], pen_opts[i])

           # lMT = torch.matmul(torch.where(moment_arm < 0, theta - np.pi, theta), moment_arm).view(batch_size, -1, 1) # MTU length, m
           pen_angle = musc.pen_angle(lMs[i], width/lMs[i]).view(batch_size, -1, 1) # pennation angle, rad
           # lT = lMT - torch.mul(lMs[i], torch.cos(pen_angle)) # tendon length, m
           # epsT = (lT - lT_slacks[i])/lT_slacks[i] # tendon strain, unitless
           lT = musc.lT(lMT, lMs[i], pen_angle) # tendon length, m
           epsT = musc.epsT(lT, lT_slacks[i]) # tendon strain, unitless

           # sanity check
           # if (lT > lMT).any(): raise ValueError(f'lT is greater than lMT, this is not possible!')

           # muscle force relationships
           tendon_force = f_maxs[i]*musc.tendon_force_N(epsT) # tendon force, N
           active_force = f_maxs[i]*Alphas[:, i].view(batch_size, -1, 1).mul(musc.active_force_N(lMs[i]/lM_opts[i])) # active muscle force, N
           passive_force = f_maxs[i]*musc.passive_force_N(lMs[i]/lM_opts[i]) # passive muscle force, N
           contractile_force = musc.contractile_element(tendon_force, pen_angle, passive_force) # contractile element force, N

           # fiber velocity
           dlM_dt_norm = musc.i_velocity_force(contractile_force, active_force) # fiber velocity normalized by maximum contraction speed, unitless
       #     dlM_dt_norm = musc.alt_i_velocity_force(contractile_force.div(active_force)) # using the alternative parameterization
           vMax = musc.vMaxCal(vM_maxs[i], vAlphas[i], Alphas[:, i])*lM_opts[i]
           vM = moment_arm.sign()*vMax.view(-1, 1, 1).mul(dlM_dt_norm) # this has units of m/s - do not modify with optimal lengths or other

           lM_out = lMs[i] + vM*dt
           lM_outs.append(musc.fiberLenConstrain(lM_out, lM_opts[i], pen_opts[i]))

           # store state information
           vMs.append(vM)
           fMs.append(tendon_force)
           lMTs.append(lMT)
           rMs.append(moment_arm)

           # make intermediates
           vals['lM'].append(lMs[i])
           vals['alpha'].append(Alphas[:, i].view(-1, 1, 1))
           vals['rMT'].append(moment_arm)
           vals['width'].append(width.expand((lMT.shape[0], -1)))
           vals['lMT'].append(lMT)
           vals['lMNorm'].append(lMs[i]/lM_opts[i])
           vals['pen'].append(pen_angle)
           vals['lT'].append(lT)
           vals['epsT'].append(epsT)
           vals['tendon'].append(tendon_force)
           vals['active'].append(active_force)
           vals['passive'].append(passive_force)
           vals['contractile'].append(contractile_force)
           vals['dlM_dt'].append(dlM_dt_norm)
           vals['vM'].append(vM)

       Joint_1dof.printVals(vals, jointID=jointID)
       theta_dt = omega
       omega_dt = (-K*(theta - np.pi/2) - B*omega - rMs[0]*fMs[0] - rMs[1]*fMs[1])/I
       #   lM0_dt = vMs[0]
       #   lM1_dt = vMs[1]
       
       #   dts = {'theta_dt': theta_dt.squeeze().detach().cpu().numpy(), 'omega_dt': omega_dt.squeeze().detach().cpu().numpy(), 'lM0_dt': lM0_dt.squeeze().detach().cpu().numpy(), 'lM1_dt': lM1_dt.squeeze().detach().cpu().numpy()}
       dts = {'theta_dt': theta_dt.squeeze().detach().cpu().numpy(), 'omega_dt': omega_dt.squeeze().detach().cpu().numpy(), 'lM0norm_dt': lM_outs[0].div(lM_opts[0]).squeeze().detach().cpu().numpy(), 'lM1norm_dt': lM_outs[1].div(lM_opts[1]).squeeze().detach().cpu().numpy()}
       print('Output:')
       print(pd.DataFrame.from_dict(dts, orient='index'))
       # print(f'Output: \n\ttheta_dt:\n\t\t{theta_dt.squeeze().detach().cpu().numpy()}\n\tomega_dt:\n\t\t{omega_dt.squeeze().detach().cpu().numpy()}\n\tlM0_dt:\n\t\t{lM0_dt.squeeze().detach().cpu().numpy()}\n\tlM1_dt:\n\t\t{lM1_dt.squeeze().detach().cpu().numpy()}')
       
       if vMs[0].isnan().any() or vMs[1].isnan().any():
           print(f'vM0s:\n\t\t{vMs[0].detach()}')
           print(f'vM1s:\n\t\t{vMs[1].detach()}')
           raise ValueError('invalid muscle fiber velocity')
      
       #   SSout = torch.hstack((theta + theta_dt*dt, omega + omega_dt*dt, lMs[0] + lM0_dt*dt, lMs[1] + lM1_dt*dt)).view(batch_size, 4)
       SSout = torch.hstack((theta + theta_dt*dt, omega + omega_dt*dt, lM_outs[0], lM_outs[1])).view(batch_size, 4)

       print(f'\tSSout:\n{SSout.transpose(0, 1).squeeze().detach().cpu().numpy()}')

       return SSout[:, 0].view(batch_size, 1), SSout

    def print_params(self):
       f_maxs = [(self.f_maxs[i]*self.f_max_scale*self.Lr_scale).abs().detach().cpu().numpy()[0]
              for i in range(self.muscle_num)]
       lM_opts = [(self.lM_opts[i]*self.lM_opt_scale*self.Lr_scale).abs().detach().cpu().numpy()[0]
              for i in range(self.muscle_num)]
       vM_maxs = [(self.vM_maxs[i]*self.vM_max_scale*self.Lr_scale).abs().detach().cpu().numpy()[0]
              for i in range(self.muscle_num)]
       vAlphas = [(self.vAlphas[i]*self.vAlpha_scale*self.Lr_scale).abs().detach().cpu().numpy()[0]
              for i in range(self.muscle_num)]
       lT_slacks = [(self.lT_slacks[i]*self.lT_slack_scale*self.Lr_scale).abs().detach().cpu().numpy()[0]
              for i in range(self.muscle_num)]
       pen_opts = [(self.pen_opts[i]*self.pen_scale*self.Lr_scale).abs().detach().cpu().numpy()[0]
              for i in range(self.muscle_num)]
       # moment_arms = [(self.moment_arms[i]*self.moment_arm_scale*self.Lr_scale).detach().cpu().numpy()[0]
       #        for i in range(self.muscle_num)]
       lIns = [(self.lIns[i]*self.lIn_scale*self.Lr_scale).detach().cpu().numpy()[0]
              for i in range(self.muscle_num)]
       lOrigs = [(self.lOrigs[i]*self.lOrig_scale*self.Lr_scale).detach().cpu().numpy()[0]
              for i in range(self.muscle_num)]
        
       I = (self.I*self.I_scale*self.Lr_scale).abs().detach().cpu().numpy()
       B = (self.B*self.B_scale*self.Lr_scale).abs().detach().cpu().numpy()
       K = (self.K*self.K_scale*self.Lr_scale).abs().detach().cpu().numpy()
       
       print(f'f_maxs:\n\t\t{f_maxs}')
       print(f'lM_opts:\n\t\t{lM_opts}')
       print(f'vM_maxs:\n\t\t{vM_maxs}')
       print(f'vAlpha:\n\t\t{vAlphas}')
       print(f'lT_slacks:\n\t\t{lT_slacks}')
       print(f'Pennation:\n\t\t{pen_opts}')
       # print(f'Moment arms:\n\t\t{moment_arms}')
       print(f'lInsertion:\n\t\t{lIns}')
       print(f'lOrigins:\n\t\t{lOrigs}')
       print(f'I:\n\t\t{I}')
       print(f'B:\n\t\t{B}')
       print(f'K:\n\t\t{K}')

    # I'm for debugging
    @staticmethod
    def printVals(vals, jointID):
       # func = lambda tensList: torch.cat(tensList, dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()
       func = lambda tensList, i: tensList[i].transpose(0, 1).squeeze().detach().cpu().numpy()
       vals0 = dict((key, func(val, 0)) for key, val in vals.items())
       vals1 = dict((key, func(val, 1)) for key, val in vals.items())

       print(f'Intermediates for joint {jointID}:')
       print(pd.DataFrame.from_dict(vals0, orient='index'))
       print(pd.DataFrame.from_dict(vals1, orient='index'))

        # print(f"Intermediates:")
        # print(f"\talpha:\n{torch.cat(vals['alpha'], dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()}\n")
        # print(f"\tmoment_arm:\n{torch.cat(vals['rMT'], dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()}\n")
        # print(f"\twidth:\n{torch.cat(vals['width'], dim=1).detach().cpu().numpy()}\n")
        # print(f"\tlMT:\n{torch.cat(vals['lMT'], dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()}\n")
        # print(f"\tlMnorm:\n{torch.cat(vals['lMNorm'], dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()}\n")
        # print(f"\tpen_angle:\n{torch.cat(vals['pen'], dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()}\n")
        # print(f"\tlT:\n{torch.cat(vals['lT'], dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()}\n")
        # print(f"\tepsT:\n{torch.cat(vals['epsT'], dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()}\n")
        # print(f"\ttendon:\n{torch.cat(vals['tendon'], dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()}\n")
        # print(f"\tactive:\n{torch.cat(vals['active'], dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()}\n")
        # print(f"\tpassive:\n{torch.cat(vals['passive'], dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()}\n")
        # print(f"\tcontractile:\n{torch.cat(vals['contractile'], dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()}\n")
        # print(f"\tdlM_dt:\n{torch.cat(vals['dlM_dt'], dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()}\n")
        # print(f"\tvM:\n{torch.cat(vals['vM'], dim=1).transpose(0, 1).squeeze().detach().cpu().numpy()}\n")

################## OLD
            # print(f'Intermediates: \n\tmoment_arm: {moment_arm.shape}\n\twidth: {width.shape}\n\tlMT: {lMT.shape}\n\tpen_angle: {pen_angle.shape}\n\tlT: {lT.shape}\n\tepsT: {epsT.shape}')
            # print(f'\ttendon: {tendon_force.shape}\n\tactive: {active_force.shape}\n\tpassive: {passive_force.shape}\n\tcontractile: {contractile_force.shape}\n\tdlM_dt: {dlM_dt_norm.shape}\n\tvM: {vM.shape}')

            # lMT = torch.matmul(theta, moment_arm)[:, 0]
            # dlMT_dt = torch.matmul(omega, moment_arm)[:, 0]

            # lMT = torch.sqrt(self.L2**2 + self.L3**2 + signMapping[i]*2*self.L2*self.L3*torch.cos(theta + math.pi/2))
            # rM = self.L2*self.L3*torch.sin(theta + math.pi/2)/lMT

            # lMTs.append(lMT)
            # rMs.append(moment_arm)

            # lT = lMT - lMs[i]*torch.cos(pen_opts[i]) # tendon length
            # lM_norm = lMs[i]/lM_opts[i] # normalized muscle length

            # epsT = (lT - lT_slacks[i])/lT_slacks[i] # tendon strain

            # tendon force
            # need to precompute these values to use torch.where()
            # fTendLin = musc.k_lin*(epsT - musc.epsT_toe) + musc.fT_toe
            # fTendExp = musc.fT_toe*(torch.exp(musc.k_toe*epsT/musc.epsT_toe) - 1)/math.exp(musc.k_toe - 1)
            # fTendZero = torch.zeros_like(fTendExp)

            # fTend = torch.where(epsT > musc.epsT_toe, fTendLin, torch.where(epsT > 0, fTendExp, fTendZero))
            # f_T = 0.001*(1 + epsT) + fTend
            # fMs.append(f_T)

            # f_AL_norm = torch.exp(-(lM_norm - 1)**2/musc.k_AL) # normalized active force
            # f_AL = f_maxs[i]*torch.mul(Alphas[:, i].view(-1, 1, 1), f_AL_norm.view(-1, 1, 1)) # actual muscle force

            # passive muscle force
            # again, precompute to use with torch.where()
            # f_PL_normLong = 1 + musc.k_PL/musc.epsM_0*(lM_norm - (1 + musc.epsM_0))
            # f_PL_normShort = torch.exp(musc.k_PL*(lM_norm - 1)/musc.epsM_0)/math.exp(musc.k_PL)
            # f_PL_norm = torch.where(lM_norm > 1 + musc.epsM_0, f_PL_normLong, f_PL_normShort)

            # f_PL = f_maxs[i]*f_PL_norm

            # contractile element force
            # f_CE = f_T - f_PL

            # inverted force-velocity relationship to find the normalized fiber velocity
            # precompute again...
            # vM_normNeg = f_CE/musc.eps*((musc.eps - f_AL)/(f_AL + musc.eps/musc.Af + musc.zeta) + f_AL/(f_AL + musc.zeta)) - f_AL/(f_AL + musc.zeta)
            # vM_normShort = (f_CE - f_AL)/(f_AL + f_CE/musc.Af + musc.zeta)
            # vM_normLong = (f_CE - f_AL)/((2 + 2/musc.Af)*(f_AL*musc.fLen_max - f_CE)/(musc.fLen_max - 1) + musc.zeta)
        
            # f_v0 = (0.95*f_AL*musc.fLen_max - f_AL)/((2 + 2/musc.Af)*0.05*f_AL*musc.fLen_max/(musc.fLen_max - 1) + musc.zeta)
            # f_v1 = ((0.95 + musc.eps)*f_AL*musc.fLen_max - f_AL)/((2 + 2/musc.Af)*(0.05 - musc.eps)*f_AL*musc.fLen_max/(musc.fLen_max - 1) + musc.zeta)
            # vM_normLongest = f_v0 + (f_CE - 0.95*f_AL*musc.fLen_max)/(musc.eps*f_AL*musc.fLen_max)*(f_v1 - f_v0)

            # vM_norm = torch.where(f_CE < 0, vM_normNeg, torch.where(f_CE < f_AL, vM_normShort, torch.where(f_CE < 0.95*f_AL*musc.fLen_max, vM_normLong, vM_normLongest))) # fiber velocity normalized by VMax

            # vM_opt_s = torch.mul(vM_maxs[i], vM_norm) # actual fiber velocity in optimal lengths per second(??)
            # vM = torch.mul(vM_opt_s, lM_opts[i]) # fiber velocity in m/s

            # if torch.any(torch.isnan(torch.cat(lMs))):
            #     print(f'Muscle{i}: lMT {lMT.detach()}, rM {rMs[i].detach()}, lT {lT.detach()}, lM_norm {lM_norm.detach()}')
            #     print(f'       : epsT {epsT.detach()}, fTend {fTend.detach()}, f_T {f_T.detach()}, f_AL {f_AL.detach()}, f_PL {f_PL.detach()}, f_CE {f_CE.detach()}')
            #     print(f'       : f_v0 {f_v0.detach()}, f_v1 {f_v1.detach()}, vM {vM.detach()}')

            # vMs.append(vM)
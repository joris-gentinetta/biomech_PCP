# Mikey Fernandez July 24 2024
"""
Hill Muscles implemented here - short rigid tendon or otherwise
"""
import math
import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization
from utils import Exponential

from torch.nn.functional import relu

## Muscle parameters
fMax = math.log(100)
lM_opt = math.log(0.06)
c = 1
A = 1
lT_s = math.log(0.04)
c_PE = math.log(10)
c_AL = math.log(7)
a_con = 0
b_con = 0
a_ecc = 0
b_ecc = 0
fEcc = math.log(1.8)

## Joint parameters
M = 0.05
I = math.log(0.004)
B = math.log(.3)
K = math.log(5)

class Muscles_Hill(nn.Module):
    """ Hill muscle model with muscle length included as a state """
    def __init__(self, device, n_joints):
        super().__init__()

        self.device = device
        self.n_joints = n_joints

        self.fMax = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * fMax)
        self.lM_opt = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * lM_opt)
        self.lT_s = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * lT_s)
        self.c_PE = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * c_PE)
        self.c_AL = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * c_AL)
        self.a_con = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * a_con)
        self.b_con = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * b_con)
        self.a_ecc = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * a_ecc)
        self.b_ecc = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * b_ecc)
        self.fEcc = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * fEcc)

        ## define constants
        self.epsT_toe = 0.0127 # strain at transition to linear
        self.ET = 37.5 # young's modulus of tendon
        self.T_affine = 0.2375 # intercept for tendon quadratic/linear
        self.quadT = 1480.3

        parameterization = Exponential()
        for parameter in ['fMax', 'lM_opt', 'lT_s', 'c_PE', 'c_AL', 'a_con', 'b_con', 'a_ecc', 'b_ecc', 'fEcc']:
            register_parametrization(self, parameter, parameterization)

    def forward(self, alphas, states):
        del_lM = states[0] # note that I'm already defined in terms of strain
        muscle_SS = states[1]

        alphas = torch.clamp(alphas, 0.001, 1) # make sure to avoid singularity issues)

        del_lMT = muscle_SS[:, :, 0, :]
        # vMT = muscle_SS[:, :, 1, :]

        epsMT = del_lMT#/self.lM_opt
        epsM = del_lM#/self.lM_opt
        epsT = (epsMT - epsM)*self.lM_opt/self.lT_s# - 1

        f_PE = self.fMax*torch.exp(self.c_PE*(epsM - 0.5))
        f_L = self.fMax*torch.exp(-self.c_AL*epsM**2)
        # f_V = nn.functional.relu(-torch.atan(-0.5*self.velM)/math.atan(5) + 1)
        f_SE = self.fMax*torch.where(epsT > self.epsT_toe, self.ET*epsT - self.T_affine, self.quadT*epsT*nn.functional.relu(epsT))

        # get the velocity scaling factor
        fV = (f_SE - f_PE)/(alphas*f_L)  # todo we can rewrite this with the damped equilibrum model, allowing us to find the unique velocity solution that maintains equilibrum
        # fV = nn.functional.relu(f_SE - f_PE)/(alphas*f_L)

        # invert the force velocity relationship to get the velocity, in terms of optimal fiber lengths per second
        # vel_opts = torch.where(fV <= 1, self.b_con*(1 - fV)/(fV + self.a_con), self.b_ecc*(fV - 1)/(self.a_ecc*(1 - self.fEcc) - self.fEcc + fV))
        vel_opts = torch.where(fV <= 0, 0, torch.where(fV <= 1, self.b_con*(1 - fV)/(fV + self.a_con), self.b_ecc*(fV - 1)/(self.a_ecc*(1 - self.fEcc) - self.fEcc + fV)))
        d_epsM_dt = vel_opts*self.lM_opt

        # if we assume this is a normalized strain rate, we should be able to adjust to the actual strain rate
        # d_epsM_dt = d_epsM_dt*self.lM_opt

        K_PE = self.c_PE/self.lM_opt*f_PE
        K_L = -2*alphas*self.c_AL*epsM*f_L/self.lM_opt # this can be negative so watch me!
        K_SE = torch.where(epsT > self.epsT_toe, self.ET/self.lT_s, 2*self.quadT*nn.functional.relu(epsT)/self.lT_s)
        # K_M = K_L*fV + K_PE
        # K_PE = self.c_PE*f_PE
        # K_L = -2*alphas*self.c_AL*epsM*f_L # this can be negative so watch me!
        # K_SE = torch.where(epsT > self.epsT_toe, self.ET, 2*self.quadT*nn.functional.relu(epsT))
        # K_M = nn.functional.relu(K_L*fV + K_PE)
        K_M = K_PE

        # muscle force is just the tendon force, which definitionally must equal the muscle force
        F = f_SE

        # stiffness is from the muscle (parallel elements) and tendon (series element)
        K = 1/(1/K_M + 1/K_SE)
        # K = K_M + K_SE

        return F, K, d_epsM_dt

    def get_starting_states(self, batch_size, y=None, x=None):
        alphas_raw = x[0]  # this is [batch_size, sequence_length, 2*n_joints]
        muscle_SS = x[1]

        alphas = alphas_raw[:, 0, :].view(-1, self.n_joints, 2, 1)  # this is now [batch_size, n_joints, 2, 1] to match other values

        steps = 20001

        # for these calculations will need to unsqueeze parameters along the range dimension
        lM_opt = self.lM_opt.unsqueeze(0).unsqueeze(3)
        lT_s = self.lT_s.unsqueeze(0).unsqueeze(3)
        c_PE = self.c_PE.unsqueeze(0).unsqueeze(3)
        c_AL = self.c_AL.unsqueeze(0).unsqueeze(3)

        del_lM_range = torch.linspace(-1, 1, steps).to(self.device) # this is defined in terms of strain - we'd never expect strain outside of this range!
        del_lM = del_lM_range.repeat(batch_size, self.n_joints, 2, 1)

        # calculate force components
        del_lMT = muscle_SS[:, :, 0, :].unsqueeze(3).repeat(1, 1, 1, steps)

        epsMT = del_lMT#/lM_opt
        epsM = del_lM#/lM_opt
        epsT = (epsMT - epsM)*lM_opt/lT_s# - 1

        f_PE = torch.exp(c_PE*(epsM - 0.5))
        f_L = torch.exp(-c_AL*epsM**2)
        # f_V = nn.functional.relu(-torch.atan(-0.5*self.velM)/math.atan(5) + 1)
        f_V = torch.ones_like(f_L)
        f_SE = torch.where(epsT > self.epsT_toe, self.ET*epsT - self.T_affine, self.quadT*epsT*nn.functional.relu(epsT))

        F_err = torch.abs(alphas*f_L*f_V + f_PE - f_SE)

        # then the lowest value of the force is the initial state
        min_idx = torch.argmin(F_err, dim=-1)
        initial_del_lM = del_lM_range[min_idx]

        # calculate force components for validation
        epsM_V = initial_del_lM#/self.lM_opt
        epsMT_V = muscle_SS[:, :, 0, :]#/self.lM_opt
        epsT_V = (epsMT_V - epsM_V)*self.lM_opt/self.lT_s# - 1

        f_PE_V = self.fMax*torch.exp(self.c_PE*(epsM_V - 0.5))
        f_L_V = self.fMax*torch.exp(-self.c_AL*epsM_V**2)
        f_SE_V = self.fMax*torch.where(epsT_V > self.epsT_toe, self.ET*epsT_V - self.T_affine, self.quadT*epsT_V*nn.functional.relu(epsT_V))

        F_V = alphas_raw[:, 0, :].view(-1, self.n_joints, 2)*f_L_V + f_PE_V
        F_err_V = torch.abs(F_V - f_SE_V)

        return initial_del_lM
    # def forward(self, alphas, states):
    #     del_lM = states[0] # note that I'm already defined in terms of strain
    #     muscle_SS = states[1]
    #
    #     alphas = torch.clamp(alphas, 0.001, 1) # make sure to avoid singularity issues)
    #
    #     del_lMT = muscle_SS[:, :, 0, :]
    #     # vMT = muscle_SS[:, :, 1, :]
    #
    #     epsMT = del_lMT/self.lM_opt
    #     epsM = del_lM/self.lM_opt
    #     epsT = (epsMT - epsM)*self.lM_opt/self.lT_s - 1
    #
    #     f_PE = self.fMax*torch.exp(self.c_PE*(epsM - 0.5))
    #     f_L = self.fMax*torch.exp(-self.c_AL*epsM**2)
    #     # f_V = nn.functional.relu(-torch.atan(-0.5*self.velM)/math.atan(5) + 1)
    #     f_SE = self.fMax*torch.where(epsT > self.epsT_toe, self.ET*epsT - self.T_affine, self.quadT*epsT*nn.functional.relu(epsT))
    #
    #     # get the velocity scaling factor
    #     # fV = (f_SE - f_PE*self.fMax)/(alphas*f_L*self.fMax)
    #     fV = (f_SE - f_PE)/(alphas*f_L)
    #
    #     # invert the force velocity relationship to get the velocity
    #     d_epsM_dt = torch.where(fV <= 1, self.b_con*(1 - fV)/(fV + self.a_con), self.b_ecc*(fV - 1)/(self.a_ecc*(1 - self.fEcc) - self.fEcc + fV))
    #
    #     K_PE = self.c_PE/self.lM_opt*f_PE
    #     K_L = -2*self.c_AL*epsM*f_L # this can be negative so watch me!
    #     K_SE = torch.where(epsT > self.epsT_toe, self.ET, 2*self.quadT*nn.functional.relu(epsT))
    #     K_M = K_L*fV + K_PE
    #
    #     # muscle force is just the tendon force, which definitionally must equal the muscle force
    #     F = f_SE
    #
    #     # stiffness is from the muscle (parallel elements) and tendon (series element)
    #     K = 1/(1/K_M + 1/K_SE)
    #
    #     return F, K, d_epsM_dt
    #
    # def get_starting_states(self, batch_size, y=None, x=None):
    #     alphas_raw = x[0]  # this is [batch_size, sequence_length, 2*n_joints]
    #     muscle_SS = x[1]
    #
    #     alphas = alphas_raw[:, 0, :].view(-1, self.n_joints, 2, 1)  # this is now [batch_size, n_joints, 2, 1] to match other values
    #
    #     steps = 20001
    #
    #     # for these calculations will need to unsqueeze parameters along the range dimension
    #     lM_opt = self.lM_opt.unsqueeze(0).unsqueeze(3)
    #     lT_s = self.lT_s.unsqueeze(0).unsqueeze(3)
    #     c_PE = self.c_PE.unsqueeze(0).unsqueeze(3)
    #     c_AL = self.c_AL.unsqueeze(0).unsqueeze(3)
    #
    #     del_lM_range = torch.linspace(-1, 1, steps).to(self.device)
    #     del_lM = del_lM_range.repeat(batch_size, self.n_joints, 2, 1)
    #
    #     # calculate force components
    #     del_lMT = muscle_SS[:, :, 0, :].unsqueeze(3).repeat(1, 1, 1, steps)
    #
    #     epsMT = del_lMT/lM_opt
    #     epsM = del_lM/lM_opt
    #     epsT = (epsMT - epsM)*lM_opt/lT_s - 1
    #
    #     f_PE = torch.exp(c_PE*(epsM - 0.5))
    #     f_L = torch.exp(-c_AL*epsM**2)
    #     # f_V = nn.functional.relu(-torch.atan(-0.5*self.velM)/math.atan(5) + 1)
    #     f_V = torch.ones_like(f_L)
    #     f_SE = torch.where(epsT > self.epsT_toe, self.ET*epsT - self.T_affine, self.quadT*epsT*nn.functional.relu(epsT))
    #
    #     F_err = torch.abs(alphas*f_L*f_V + f_PE - f_SE)
    #
    #     # then the lowest value of the force is the initial state
    #     min_idx = torch.argmin(F_err, dim=-1)
    #     initial_del_lM = del_lM_range[min_idx]
    #
    #     # calculate force components for validation
    #     epsM_V = initial_del_lM/self.lM_opt
    #     epsMT_V = muscle_SS[:, :, 0, :]/self.lM_opt
    #     epsT_V = (epsMT_V - epsM_V)*self.lM_opt/self.lT_s - 1
    #
    #     f_PE_V = self.fMax*torch.exp(self.c_PE*(epsM_V - 0.5))
    #     f_L_V = self.fMax*torch.exp(-self.c_AL*epsM_V**2)
    #     f_SE_V = self.fMax*torch.where(epsT_V > self.epsT_toe, self.ET*epsT_V - self.T_affine, self.quadT*epsT_V*nn.functional.relu(epsT_V))
    #
    #     F_V = alphas_raw[:, 0, :].view(-1, self.n_joints, 2)*f_L_V + f_PE_V
    #     F_err_V = torch.abs(F_V - f_SE_V)
    #
    #     return initial_del_lM

    # code:
    # range over muscle lengths
    # calculate forces
    # select value that minimize difference
    # return this as initial state

class Muscles_SRT(nn.Module):
    """ this is the short rigid tendon hill muscle model without a pennation angle"""
    def __init__(self, device, n_joints):
        super().__init__()

        self.device = device

        self.fMax = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * fMax)
        self.lM_opt = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * lM_opt)
        self.c = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * c)
        self.A = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * A)

        # self.grad = lambda epsM, epsMT, velM, alphas: torch.stacK([-10*torch.exp(-100*(epsMT - epsM)) + 8*alphas*epsM*(torch.atan(0.5*velM)/math.atan(5) + 1),
        #                                                               -0.5*alphas*(1 - 4*epsM**2)/(math.atan(5)*(0.25*velM**2 + 1))], dim=1)
        # self.hessian = lambda epsM, epsMT, velM, alphas: torch.stack([torch.stack([1000*torch.exp(-100*(epsMT - epsM)) + 8*alphas*(torch.atan(0.5*velM)/math.atan(5) + 1),
        #                                                                               4*alphas*epsM/(math.atan(5)*(0.25*velM**2 + 1))], dim=1),
        #                                                                 torch.stack([4*alphas*epsM/(math.atan(5)*(0.25*velM**2 + 1)),
        #                                                                              0.25*alphas*velM*(1 - 4*epsM**2)/(math.atan(5)*(0.25*velM**2 + 1)**2)], dim=1)], dim=2)

        # self.epsM = torch.zeros_like(self.fMax)
        # self.velM = torch.zeros_like(self.fMax)
        self.epsM = None
        self.velM = None

        parameterization = Exponential()
        for parameter in ['fMax', 'lM_opt', 'c', 'A']:
            register_parametrization(self, parameter, parameterization)

    def forward(self, alphas, muscle_SS):
        # alphas = torch.clamp(alphas, 0, 1)
        alphas = torch.clamp(alphas, 0.01, 1)

        if self.epsM is None:
            self.epsM = torch.zeros_like(muscle_SS[:, :, 0, :])
            self.velM = torch.zeros_like(muscle_SS[:, :, 1, :])

        lMT = muscle_SS[:, :, 0, :]
        epsMT = lMT/self.lM_opt# - 1

        for _ in range(10):
            grad = torch.stack([-10*torch.exp(-100*(epsMT - self.epsM))*(epsMT > self.epsM).float() - 8*alphas*self.epsM*(-4*self.epsM**2 + 1 > 0).float()*relu(-torch.atan(0.5*self.velM)/math.atan(5) + 1),
                                                                      -0.5*alphas*relu(1 - 4*self.epsM**2)/(math.atan(5)*(0.25*self.velM**2 + 1))*(self.velM >= -10).float()], dim=-1)
            hessian = torch.stack([torch.stack([1000*torch.exp(-100*(epsMT - self.epsM))*(epsMT > self.epsM).float() - 8*alphas*relu(-torch.atan(0.5*self.velM)/math.atan(5) + 1),
                                                                                      4*alphas*self.epsM*(-4*self.epsM**2 + 1 > 0).float()/(math.atan(5)*(0.25*self.velM**2 + 1)*(self.velM >= -10).float())], dim=-1),
                                                                        torch.stack([4*alphas*self.epsM*(-4*self.epsM**2 + 1 > 0).float()/(math.atan(5)*(0.25*self.velM**2 + 1)*(self.velM >= -10).float()),
                                                                                     0.25*alphas*self.velM*(self.velM >= -10).float()*relu(1 - 4*self.epsM**2)/(math.atan(5)*(0.25*self.velM**2 + 1)**2)], dim=-1)], dim=2)
            hessian = hessian + 1e-4*torch.eye(2, device=self.device).repeat(hessian.shape[0], hessian.shape[1], 2, 1, 1)

            update = -torch.linalg.solve(hessian, grad)
            self.epsM = self.epsM + update[..., 0]
            self.velM = self.velM + update[..., 1]

            stopCond = torch.linalg.norm(update, dim=-1)

            if torch.all(stopCond < 1e-6):
                break

        fPE = nn.functional.relu(2*self.c*self.A*epsMT*torch.exp(self.c * epsMT**2))
        fL = nn.functional.relu(-4*self.epsM**2 + 1) # defined based on deviation
        fV = nn.functional.relu(-torch.atan(-0.5*self.velM)/math.atan(5) + 1)
        fSE = nn.functional.relu(0.1*(torch.exp(100*(epsMT - self.epsM)) - 1))

        # F = self.fMax.unsqueeze(0)*(alphas*fL*fV + fPE)
        F = self.fMax.unsqueeze(0)*(fSE + fPE)

        K = self.fMax/self.lM_opt*(10*torch.exp(100*(epsMT - self.epsM))*(fSE > 0).float() + (2*self.c*self.A*torch.exp(self.c*epsMT**2) + 4*self.c**2*self.A*epsMT**2*torch.exp(self.c*epsMT**2)*(fPE > 0).float()))

        # lM = (muscle_SS[:, :, 0, :] - self.lT)
        # lM_norm = lM / self.lM_opt
        #
        # vM = -muscle_SS[:, :, 1, :]
        # vM_norm = vM / self.lM_opt # in optimal muscle lengths per second
        #
        # l_norm = muscle_SS[:, :, 0, :] / self.lM_opt
        # lSE = muscle_SS[:, :, 0, :] / lM
        #
        # fL = nn.functional.relu(-4*lM_norm**2 + 1) # defined based on deviation
        # fV = nn.functional.relu(-torch.atan(-0.5*vM_norm)/math.atan(5) + 1)
        # fPE = nn.functional.relu(2*c*A*l_norm*torch.exp(c * l_norm**2))
        # # fSE = nn.functional.relu(0.1*(torch.exp(100*lM_norm*(lSE - 1)) - 1))
        #
        # F = self.fMax.unsqueeze(0)*(alphas*fL*fV + fPE)
        #
        # K = self.fMax*(2*self.c*self.A/self.lM_opt*torch.exp(self.c * (l_norm - 1)**2) * (1 + 2*self.c*(l_norm - 1)**2) * (l_norm > 1).float()
        #         + 1000/self.lM_opt*torch.exp(100*lM_norm*(lSE - 1))*(2*lSE - 1)*(lSE > 1).float())

        return F, K


class Joints(nn.Module):
    def __init__(self, device, n_joints, dt, speed_mode=True):
        super().__init__()

        self.device = device
        self.n_joints = n_joints
        self.dt = dt
        self.speed_mode = speed_mode

        self.I = nn.Parameter(data=torch.ones(self.n_joints, dtype=torch.float, device=self.device) * I)
        self.B = nn.Parameter(data=torch.ones(self.n_joints, dtype=torch.float, device=self.device) * B)
        self.K = nn.Parameter(data=torch.ones(self.n_joints, dtype=torch.float, device=self.device) * K)

        moment_arms = torch.ones((self.n_joints, 2), dtype=torch.float, device=self.device) * M
        moment_arms[:, 0] = moment_arms[:, 0] * -1  # note that sign flipped
        self.M = nn.Parameter(data=moment_arms)

        parameterization = Exponential()
        for parameter in ['I', 'B', 'K']:
            register_parametrization(self, parameter, parameterization)

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
        A00 = torch.zeros((batch_size, self.n_joints), dtype=torch.float, device=self.device)  # [batch_size, n_joints]
        A01 = torch.ones((batch_size, self.n_joints), dtype=torch.float, device=self.device)

        K = K_musc * self.M.unsqueeze(0) * self.M.unsqueeze(0)  # [batch_size, n_joints, muscle_num]
        K = K.sum(dim=2) + self.K.unsqueeze(0)  # sum over muscles - > [batch_size, n_joints]
        A10 = -K / self.I.unsqueeze(0)  # [batch_size, n_joints]
        # A10 = -K / self.I.unsqueeze(0)  # [batch_size, n_joints]

        #############################
        #   DAMPING                 #
        #############################
        D = torch.sqrt(K * self.I.unsqueeze(0)) * 2
        A11 = -D * self.B.unsqueeze(0) / self.I.unsqueeze(0)  # with joint damping
        # A11 = -D / self.I.unsqueeze(0)  # with joint damping

        A = torch.stack([torch.stack([A00, A01], dim=2), torch.stack([A10, A11], dim=2)],
                        dim=2)  # [batch_size, n_joints, 2, 2]

        #########################
        #   MATRIX B            #
        #########################
        B0 = torch.zeros(batch_size, self.n_joints, 1, dtype=torch.float, device=self.device)

        B_F = F

        # The following K is respect to w(angle)
        B1 = B_F * self.M.unsqueeze(0) / self.I.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1)
        B1 = B1.sum(dim=2).unsqueeze(2)  # sum over muscles - > [batch_size, n_joints]

        B = torch.stack([B0, B1], dim=2)

        if not self.speed_mode:
            #############################
            #   Accurate Simulation     #
            #############################

            expAt = torch.matrix_exp(A * self.dt)
            eye = torch.eye(2, dtype=torch.float, device=self.device).unsqueeze(0).unsqueeze(0).repeat(batch_size,
                                                                                                       self.n_joints, 1,
                                                                                                       1)

            SSout = torch.matmul(expAt, SS.unsqueeze(3)) + torch.matmul(torch.linalg.solve(A, expAt - eye), B)
            SSout = SSout.squeeze(3)

        else:
            #############################
            #   Simplified Simulation   #
            #############################
            # The simplified simulation addition instead of integration
            # xdot = A x + B u
            SSout = ((torch.matmul(A, SS.unsqueeze(3)) + B) * self.dt + SS.unsqueeze(3)).squeeze(3)

        muscle_SSs = SSout.unsqueeze(3) * self.M.unsqueeze(0).unsqueeze(
            2)  # [batch_size, n_joints, state_num (2), muscle_num (2)]

        if torch.any(torch.isnan(SSout)):
            print('nan in SSout')
            print(SSout)
            print('nan in muscle_SSs')
            print(muscle_SSs)

        return SSout[:, :, 0], muscle_SSs, SSout  # position, muscle state, [position, velocity]
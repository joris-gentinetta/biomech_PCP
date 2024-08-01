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
fMax = math.log(1000)
lM_opt = math.log(0.6)
c = 1
A = 1
lT_s = 0.04
c_PE = 3
c_SE = 7

## Joint parameters
M = 0.05
I = math.log(0.004)
B = math.log(.3)
K = math.log(5)

class Muscle_Hill(nn.Module):
    """ Hill muscle model with muscle lenth included as a state """
    def __init__(self, device, n_joints):
        super().__init__()

        self.device = device

        self.fMax = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * fMax)
        self.lM_opt = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * lM_opt)
        self.lT_s = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * lT_s)
        self.c_PE = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * c_PE)
        self.c_SE = nn.Parameter(data=torch.ones((n_joints, 2), dtype=torch.float, device=self.device) * c_SE)

        ## define constants
        self.epsT_toe = 0.0127 # strain at transition to linear
        self.ET = 37.5 # young's modulus of tendon
        self.T_affine = 0.2375 # intercept for tendon quadratic/exponential
        self.quadT = 1480.3

        parameterization = Exponential()
        for parameter in ['fMax', 'lM_opt', 'lT_s', 'c_PE', 'c_SE']:
            register_parametrization(self, parameter, parameterization)

    def forward(self, alphas, states):
        del_lM = states[0]
        muscle_SS = states[1]

        alphas = torch.clamp(alphas, 0, 1)

        del_lMT = muscle_SS[:, :, 0, :]
        vMT = muscle_SS[:, :, 1, :]

        epsMT = del_lMT/self.lM_opt
        epsM = del_lM/self.lM_opt
        epsT = (epsMT - epsM)*self.lM_opt/self.LT_s - 1

        f_PE = torch.exp(self.c_PE*(epsM - 0.5))
        f_L = torch.exp(-self.c_SE*epsM**2)
        # f_V = nn.functional.relu(-torch.atan(-0.5*self.velM)/math.atan(5) + 1)
        f_SE = torch.where(epsT > self.epsT_toe, self.ET*epsT - self.T_affine, self.quadT*epsT*nn.functional.relu(epsT))

        K_PE = self.c_PE/self.lM_opt*f_PE
        K_L = -2*self.c_SE*epsM*f_L # this can be negative so watch me!
        K_SE = torch.where(epsT > self.epsT_toe, self.ET, 2*self.quadT*nn.functional.relu(epsT))

        return F, K

    def get_starting_states(self, batch_size, y=None, x=None):
        alphas = x[0]
        muscle_SS = x[1]

        del_lM_range = torch.linspace(-1, 1, 1000).repeat(batch_size, self.n_joints, 2).to(self.device)

        # calculate force components
        del_lMT = muscle_SS[:, :, 0, :]

        epsMT = del_lMT/self.lM_opt
        epsM_range = del_lM_range/self.lM_opt
        epsT = (epsMT - epsM_range)*self.lM_opt/self.LT_s - 1

        f_PE = torch.exp(self.c_PE*(epsM_range - 0.5))
        f_L = torch.exp(-self.c_SE*epsM_range**2)
        # f_V = nn.functional.relu(-torch.atan(-0.5*self.velM)/math.atan(5) + 1)
        f_V = torch.ones_like(f_L)
        f_SE = torch.where(epsT > self.epsT_toe, self.ET*epsT - self.T_affine, self.quadT*epsT*nn.functional.relu(epsT))

        F = alphas*(f_L*f_V + f_PE) - f_SE

        # then the lowest value of the force is the initial state
        min_idx = torch.argmin(F, dim=0)
        initial_del_lM = del_lM_range[torch.arange(batch_size), min_idx]

        return initial_del_lM

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
                break                                                                                                                                                                                                                                                       ak

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
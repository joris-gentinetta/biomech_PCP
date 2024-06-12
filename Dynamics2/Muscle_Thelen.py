# Mikey Fernandez 04/02/2022
"""
This class simulates the first order dynamic of one muscle based on a Thelen 2003 Muscle model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# for debugging
import inspect

class Muscle():
    # def __init__(self, f_max, lM_opt, vM_max, lT_slack, pen_opt, moment_arm):
    def __init__(self, f_max, lM_opt, velocities, lT_slack, pen_opt, lengths):
        
        """Muscle in the simulation model

        Args:
            f_max (float): maximum isometric muscle force, N
            lM_opt (float): optimal muscle fiber length, m
            vM_max (float): maximum muscle velocty, lenOpt-s^-1
            lT_slack (float): tendon slack length, m
            pen_opt (float): pennation angle at optimal fiber length, rad
            moment_arm (float): muscle moment arm, m
        """
        
        super().__init__()
        self.EPS = torch.tensor(torch.finfo(torch.float).eps)

        # define muscle parameters that will be optimized
        self.f_max = f_max
        self.lM_opt = lM_opt
        self.vM_max = velocities[0]
        self.vAlpha = velocities[1]
        self.lT_slack = lT_slack
        self.pen_opt = pen_opt
        # self.moment_arm = np.array(moment_arm)
        self.lIn = lengths[0]
        self.lOrig = lengths[1]

        # define some muscle parameters (fixed for all muscles)
        self.epsM_0 = 0.6 # passive muscle strain due to maximum isometric force
        self.k_PL = 5 # shape factor for passive force-length relationship
        self.k_AL = 0.45 # shape factor for active force-length relationship
        self.fLen_max = 1.4 # maximum normalized force achieved when muscle is lengthening
        self.k_toe = 3 # tendon exponential shape factor
        self.Af = 0.25 # force-velocity shape factor
        self.fT_toe = 33/100 # normalized tendon force at transition to linear regime
        self.epsT_0 = 0.04 # tendon strain due to maximum isometric force
        # self.k_lin = 1.712/self.epsT_0 # slope of linear tendon force-length curve
        # self.epsT_toe = 0.609*self.epsT_0 # tendon strain at transition to linear regime
        self.epsT_toe = 99*self.epsT_0*np.exp(3)/(166*np.exp(3) - 67) # updated values from OpenSim implementation
        self.k_lin = 67/(100*(self.epsT_0 - self.epsT_toe)) # updated values from OpenSim implementation

        # numbers to prevent numerical issues
        self.eps = 1e-6 # small value to prevent numerical issues
        self.zeta = 0.05 # passive damping in force-velocity relationship

        # constraint parameters
        self.minFiberLenMul = 0.01 # minimum length of fiber (relative to optimal fiber length)
        self.fV_linear = 0.95 # threshold at which force velocity curve goes linear
        self.maxPen = np.arccos(0.1)

        # functions for finding initial conditions or solving muscle state equations
        self.active_force_N = lambda lM_norm: (-(lM_norm - 1).square()/self.k_AL).exp()

        passiveLong = lambda lM_norm: 1 + self.k_PL/self.epsM_0*(lM_norm - (1 + self.epsM_0))
        passiveShort = lambda lM_norm: (self.k_PL*(lM_norm - 1)/self.epsM_0).exp()/np.exp(self.k_PL)
        self.passive_force_N = lambda lM_norm: torch.where(lM_norm > (1 + self.epsM_0), passiveLong(lM_norm), passiveShort(lM_norm))
        # passive = lambda lM_norm: ((self.k_PL*(lM_norm - 1)/self.epsM_0).exp() - 1)/(np.exp(self.k_PL) - 1)
        # self.passive_force_N = lambda lM_norm: torch.where(lM_norm > 1.0, passive(lM_norm), torch.zeros_like(lM_norm))

        tendonLin = lambda epsT: self.k_lin*(epsT - self.epsT_toe) + self.fT_toe
        tendonExp = lambda epsT: self.fT_toe*((self.k_toe*epsT/self.epsT_toe).exp() - 1)/(np.exp(self.k_toe) - 1)
        tendonNotSlack = lambda epsT: 0.001*(1 + epsT)
        self.tendon_force_N = lambda epsT: torch.where(epsT > self.epsT_toe, tendonLin(epsT), torch.where(epsT > 0, tendonExp(epsT), torch.where(epsT > -1, tendonNotSlack(epsT), torch.zeros_like(epsT))))

        neg = lambda lM: torch.zeros_like(lM)
        pi = lambda ratio: np.pi/2*torch.ones_like(ratio)
        asin = lambda ratio: ratio.asin()
        self.pen_angle = lambda lM, ratio: torch.where(lM < 0.0, neg(lM), torch.where(ratio < 0.0, neg(lM), torch.where(ratio >= 1.0, pi(ratio), asin(ratio))))

        velocityShortFast = lambda v: (v + 1)/(1 + 1/self.Af)
        velocityShortSlow = lambda v: (v + 1)/(1 - v/self.Af)
        velocityLongSlow = lambda v: ((2 + 2/self.Af)*v*self.fLen_max + self.fLen_max - 1)/((2 + 2/self.Af)*v + self.fLen_max - 1)
        velocityLongFast = lambda v: self.fLen_max/(20*(self.fLen_max - 1))*((1 + 1/self.Af)*self.fLen_max*v/(10*(self.fLen_max - 1)) + 18.05*self.fLen_max - 18)
        self.velocity_force_N = lambda v: torch.where(v < -1, velocityShortFast(v), torch.where(v < 0, velocityShortSlow(v), torch.where(v < 10*(self.fLen_max - 1)*(0.95*self.fLen_max - 1)/((1 + 1/self.Af)*self.fLen_max), velocityLongSlow(v), velocityLongFast(v))))
        
        f_v0 = lambda Fa: (0.95*Fa*self.fLen_max - Fa).div(((2 + 2/self.Af)*0.005*Fa*self.fLen_max)/(self.fLen_max - 1) + self.zeta)
        f_v1 = lambda Fa: ((0.95 + self.eps)*Fa*self.fLen_max - Fa).div(((2 + 2/self.Af)*(0.005 - self.eps)*Fa*self.fLen_max)/(self.fLen_max - 1) + self.zeta)
        fCE_neg = lambda Fce, Fa: (Fce/self.eps).mul((self.eps - Fa).div(Fa + self.eps/self.Af + self.zeta) + Fa.div(Fa + self.zeta)) - Fa.div(Fa + self.zeta)
        fCE_less_Fa = lambda Fce, Fa: (Fce - Fa).div(Fa + Fce/self.Af + self.zeta)
        fCE_less_95FaFlen = lambda Fce, Fa: (Fce - Fa).div((2 + 2/self.Af)*(Fa*self.fLen_max - Fce)/(self.fLen_max - 1) + self.zeta)
        fCE_greater = lambda Fce, Fa: f_v0(Fa) + ((Fce - 0.95*Fa*self.fLen_max).div(self.eps*Fa*self.fLen_max)).mul(f_v1(Fa) - f_v0(Fa))
        ## NOTE each of the 4 below functions are the derivatives of the above with respect to Fce
        dfCE_neg_dFce = lambda Fce, Fa: ((self.eps - Fa).div(Fa + self.eps/self.Af + self.zeta) + Fa.div(Fa + self.zeta))/self.eps
        dfCE_less_Fa_dFce = lambda Fce, Fa: (Fa + Fce/self.Af + self.zeta - (Fce - Fa)/self.Af).div((Fa + Fce/self.Af + self.zeta).square())
        dfCE_less_95FaFlen_dFce = lambda Fce, Fa: (((2 + 2/self.Af)*(Fa*self.fLen_max - Fce)/(self.fLen_max - 1) + self.zeta) - (Fce - Fa)*(2 + 2/self.Af)/(self.fLen_max - 1)).div(((2 + 2/self.Af)*(Fa*self.fLen_max - Fce)/(self.fLen_max - 1) + self.zeta).square())
        dfCE_greater_dFce = lambda Fce, Fa: (f_v1(Fa) - f_v0(Fa)).div(self.eps*Fa*self.fLen_max)
        self.i_velocity_force = lambda Fce, Fa: torch.where(Fce < 0, fCE_neg(Fce, Fa), torch.where(Fce < Fa, fCE_less_Fa(Fce, Fa), torch.where(Fce < self.fV_linear*Fa*self.fLen_max, fCE_less_95FaFlen(Fce, Fa), fCE_greater(Fce, Fa))))

        self.contractile_element = lambda fT, pen, fPE: fT.div(pen.cos()) - fPE

        self.fiber_velocity = lambda velMTU, velT, pen: (velMTU - velT).mul(pen.cos())

        # self.lMT = lambda theta, moment_arm: torch.where(moment_arm < 0, theta - np.pi, theta)*moment_arm
        self.width = lambda lM_opt, pen_opt: lM_opt*pen_opt.sin()
        self.lMT = lambda theta, lIn, lOrig: (lIn**2 + lOrig**2 - 2*lIn*lOrig*theta.cos()).sqrt() # these are explicitly calculated MTU length
        self.rMT = lambda theta, lIn, lOrig: lIn*lOrig*theta.sin().div(self.lMT(theta, lIn, lOrig)) # and moment arm
        self.drMT_dt = lambda theta, omega, lIn, lOrig: ((lIn*lOrig*theta.cos() - self.rMT(theta, lIn, lOrig).square()).mul(omega)).div(self.lMT(theta, lIn, lOrig))
        # self.dlMT_dt = lambda theta, omega, lIn, lOrig: self.drMT_dt(theta, omega, lIn, lOrig).mul(theta) + self.rMT(theta, lIn, lOrig).mul(omega)
        self.dlMT_dt = lambda theta, omega, lIn, lOrig: self.rMT(theta, lIn, lOrig).mul(omega)
        self.lT = lambda lMT, lM, pen_angle: lMT - lM.mul(pen_angle.cos())
        # self.epsT = lambda lT, lT_slack, lM_opt: (lT/lM_opt - lT_slack)/lT_slack
        self.epsT = lambda lT, lT_slack: lT/lT_slack - 1

        self.vMaxCal = lambda vM_max, vAlpha, alpha: vM_max + vAlpha*alpha

        # Just for testing [this is the inverse felocity force curve taking in just a scalar factor - no numerical stability parameters]
        f_neg = lambda f: (1 + 1/self.Af)*f - 1
        f_less1 = lambda f: (f - 1)/(1 + f/self.Af)
        f_less_95Flen = lambda f: (f - 1)*(self.fLen_max - 1)/((2 + 2/self.Af)*(self.fLen_max - f))
        f_greater = lambda f: 10*(self.fLen_max - 1)/((1 + 1/self.Af)*self.fLen_max)*(-18.05*self.fLen_max + 18 + 20*f*(self.fLen_max - 1)/self.fLen_max)
        self.alt_i_velocity_force = lambda f: torch.where(f < 0, f_neg(f), torch.where(f < 1, f_less1(f), torch.where(f < self.fV_linear*self.fLen_max, f_less_95Flen(f), f_greater(f))))

        # for finding initial conditions
        self.fiberLen = lambda lMT, lT: torch.where(lMT - lT >= self.minFiberLenMul*self.lM_opt*np.cos(self.maxPen), (self.lM_opt**2*np.sin(self.pen_opt)**2 + (lMT - lT).square()).sqrt(), self.minFiberLenMul*self.lM_opt)

        # constraint equations
        self.maxFiberLen = lambda lM_opt, pen_opt: self.width(lM_opt, pen_opt)/np.sin(self.maxPen)
        # self.fiberLenConstrain = lambda lM, lM_opt, pen_opt: torch.where(lM <= self.minFiberLenMul*lM_opt, self.minFiberLenMul*lM_opt, torch.where(lM >= self.maxFiberLen(lM_opt, pen_opt), self.maxFiberLen(lM_opt, pen_opt), lM)) # gives a minimum fiber length
        self.fiberLenConstrain = lambda lM, lM_opt, pen_opt: torch.where(lM <= self.minFiberLenMul*lM_opt, self.minFiberLenMul*lM_opt, lM) # gives a minimum fiber length

        # partial derivative functions
        self.dcosPen_dlM = lambda lM, w: w.square().div(lM.pow(3).mul((1 - (w.square().div(lM.square()))).sqrt()))
        self.dfM_dlM = lambda lM, alpha, fV: (alpha.mul(fV).mul(self.dfA_dlM(lM/self.lM_opt)) + self.dfPE_dlM(lM/self.lM_opt))*self.f_max/self.lM_opt
        self.dfMang_dlM = lambda lM, alpha, fV, pen, w, fM: self.dfM_dlM(lM, alpha, fV).mul(pen.cos()) + fM.mul(self.dcosPen_dlM(lM, w))
        self.dfMang_dlMang = lambda lM, alpha, fV, pen, w, fM: self.dfMang_dlM(lM, alpha, fV, pen, w, fM).mul(pen.cos())
        self.dPen_dlM = lambda w, lM: -w.div(lM.square().mul((1 - (w/lM).square()).sqrt()))
        self.dlT_dlM = lambda lM, pen, w: -pen.cos() + lM.mul(pen.sin()).mul(self.dPen_dlM(w, lM))
        self.dfT_dlT = lambda lT: self.f_max/self.lT_slack*self.dfT_dlTN(self.epsT(lT, self.lT_slack))
        self.dfT_dlM = lambda lT, lM, pen, w: self.f_max/self.lT_slack*self.dfT_dlTN(self.epsT(lT, self.lT_slack)).mul(self.dlT_dlM(lM, pen, w))
        self.dfT_dlTN = lambda epsT: torch.where(epsT > self.epsT_toe, self.k_lin, torch.where(epsT > 0, self.fT_toe/(np.exp(self.k_toe) - 1)*self.k_toe/self.epsT_toe*(self.k_toe*epsT/self.epsT_toe).exp(), 0))
        self.dfA_dlM = lambda lM_norm: (-2*(lM_norm - 1)/self.k_AL).mul(self.active_force_N(lM_norm))
        self.dfPE_dlM = lambda lM_norm: torch.where(lM_norm > 1.0, self.k_PL/self.epsM_0*(self.k_PL*(lM_norm - 1)/self.epsM_0).exp()/(np.exp(self.k_PL) - 1), torch.zeros_like(lM_norm))
        self.dvM_dFM = lambda Fa, Fce: torch.where(Fce < 0, dfCE_neg_dFce(Fce, Fa), torch.where(Fce < Fa, dfCE_less_Fa_dFce(Fce, Fa), torch.where(Fce < self.fV_linear*Fa*self.fLen_max, dfCE_less_95FaFlen_dFce(Fce, Fa), dfCE_greater_dFce(Fce, Fa))))

    def plotMuscleCurves(self, lM_norm_range=[0, 1.75], v_range=[-1, 1], eps_range=[0, 0.05], f_range=[-.25, 1.5], steps=2000, numPlots=3):
        lM = torch.linspace(lM_norm_range[0], lM_norm_range[1], steps=steps)
        v = torch.linspace(v_range[0], v_range[1], steps=steps)
        epsT = torch.linspace(eps_range[0], eps_range[1], steps=steps)
        f = torch.linspace(f_range[0], f_range[1], steps=steps)

        tendon_force = self.tendon_force_N(epsT) # tendon force
        active_force = self.active_force_N(lM) # active force-length relationship
        passive_force = self.passive_force_N(lM) # passive force-lenth relationship
        velocity_force = self.velocity_force_N(v) # force-velocity relationship
        i_velocity_force = self.alt_i_velocity_force(f) # inverse force-velocity curve

        plt.clf()
        plt.figure(1)
        plt.subplot(1, numPlots, 1)
        plt.plot(lM.squeeze().numpy(), active_force.squeeze().numpy())
        plt.plot(lM.squeeze().numpy(), passive_force.squeeze().numpy())
        plt.plot(lM.squeeze().numpy(), (active_force + passive_force).squeeze().numpy())
        plt.title('Fiber Force-Length')
        plt.xlabel('Normalized Length')
        plt.ylabel('Normalized Force')

        plt.subplot(1, numPlots, 2)
        plt.plot(v.squeeze().numpy(), velocity_force.squeeze().numpy())
        # plt.plot(i_velocity_force.squeeze().numpy(), f.squeeze().numpy())
        plt.title('Fiber Force-Velocity')
        plt.xlabel('Normalized Velocity')
        plt.ylabel('Normalized Force')

        plt.subplot(1, numPlots, 3)
        plt.plot(epsT.squeeze().numpy(), tendon_force.squeeze().numpy())
        plt.title('Tendon Force-Length')
        plt.xlabel('Tendon Strain')
        plt.ylabel('Normalized Force')

        if numPlots == 4:
            plt.subplot(1, numPlots, 4)
            plt.plot(f.squeeze().numpy(), i_velocity_force.squeeze().numpy())
            plt.title('Fiber Inverse Force-Velocity')
            plt.xlabel('Normalized Force')
            plt.ylabel('Normalized Velocity')

        plt.suptitle('Thelen Muscle Constitutive Relationships')
        plt.show()
    
    def findInitlM(self, theta, omega, alpha, steps=4000, storing=False):
        lMRaw = torch.linspace(self.minFiberLenMul*self.lM_opt, 2*self.lM_opt, steps, dtype=torch.float, device=theta.device)
        lM = lMRaw.expand(theta.shape[0], steps) # this should be [batchSize x steps], now we can test for each row

        if storing:
            forceErrs, storage = self.equilibrateMuscle(lM, theta, omega, alpha, storage=dict())
            _, idx = torch.min(forceErrs.abs(), dim=1)

            return lMRaw[idx], storage, idx
        
        else:
            forceErrs = self.equilibrateMuscle(lM, theta, omega, alpha)
            _, idx = torch.min(forceErrs.abs(), dim=1)

            return lMRaw[idx]

    def equilibrateMuscle(self, lM, theta, omega, alpha, storage=None):
        # print(self.lIn, self.lOrig, self.pen_opt, self.lM_opt)
        numSteps = lM.shape[1]

        theta = theta.expand(numSteps, -1).transpose(0, 1)
        omega = omega.expand(numSteps, -1).transpose(0, 1)
        alpha = alpha.expand(numSteps, -1).transpose(0, 1)

        lMT = self.lMT(theta, self.lIn, self.lOrig)
        moment_arm = self.rMT(theta, self.lIn, self.lOrig)
        
        # dlMT_dt = omega.mul(moment_arm) # velocity of MTU
        dlMT_dt = self.dlMT_dt(theta, omega, self.lIn, self.lOrig)
        pen_angle = self.pen_angle(lM, ((self.lM_opt*np.sin(self.pen_opt))/lM)) # pennation angle
        lT = self.lT(lMT, lM, pen_angle)
        epsT = self.epsT(lT, self.lT_slack)

        tendon_force = self.tendon_force_N(epsT) # tendon force
        active_force = self.active_force_N(lM/self.lM_opt) # active force-length relationship
        passive_force = self.passive_force_N(lM/self.lM_opt) # passive force-lenth relationship
        velocity_force = self.velocity_force_N(dlMT_dt.div(self.lM_opt*self.vMaxCal(self.vM_max, self.vAlpha, alpha))) # force-velocity relationship
        # we assume that because tendon is much stiffer than muscle, MTU velocity is muscle fiber velocity, though it still needs to be normalized properly

        forceErr = (alpha.mul(active_force).mul(velocity_force) + passive_force).mul(pen_angle.cos()) - tendon_force

        ## Do some stuff for validation sake down here
        fCE = self.contractile_element(tendon_force, pen_angle, passive_force)
        # fV_inv = self.i_velocity_force(fCE, active_force)
        fV_inv = self.alt_i_velocity_force(fCE.div(active_force))
        fiberVel = fV_inv.mul(self.lM_opt*self.vMaxCal(self.vM_max, self.vAlpha, alpha))

        if storage is not None:
            storage['theta'] = theta
            storage['lM'] = lM
            storage['lMT'] = lMT
            storage['rMT'] = moment_arm
            storage['dlMT_dt'] = dlMT_dt
            # storage['pen_angle'] = pen_angle
            storage['lT'] = lT
            storage['epsT'] = epsT
            storage['tendon'] = tendon_force
            # storage['active'] = active_force
            # storage['passive'] = passive_force
            storage['fV_inv'] = fV_inv
            storage['velocity'] = velocity_force
            storage['forceErr'] = forceErr
            storage['fiberVel'] = fiberVel

            return forceErr, storage

        return forceErr # call this err because we want this number driven to 0
    
    def newtonInit(self, theta, omega, alpha, tol=1e-4, maxIters=1000):
        eps = self.EPS.to(device=theta.device, dtype=theta.dtype)
        h = torch.ones_like(theta) # step size, effectively

        iters = 0; innerIters = 0

        w = torch.tensor(self.lM_opt*np.sin(self.pen_opt)).to(device=theta.device, dtype=theta.dtype) # use me throughout

        # helper functions, as given in the OpenSim code
        positionFunc = lambda lM, lMT: [self.pen_angle(lM, w), lMT - lM.mul(self.pen_angle(lM, w).cos()), lM/self.lM_opt, (lMT - lM.mul(self.pen_angle(lM, w).cos()))/self.lT_slack] # returns lT, lM_norm, lT_norm
        multipliersFunc = lambda lM_norm, lT_norm: [self.active_force_N(lM_norm), self.passive_force_N(lM_norm), self.tendon_force_N(lT_norm - 1)]
        fMFunc = lambda alpha, active_force, passive_force, velocity_force: (alpha.mul(active_force).mul(velocity_force) + passive_force)*self.f_max
        fErrFunc = lambda alpha, active_force, passive_force, velocity_force, tendon_force, pen: fMFunc(alpha, active_force, passive_force, velocity_force).mul(pen.cos()) - tendon_force*self.f_max
        partialsFunc = lambda alpha, lM, fV, pen, lT, w, fM: [self.dfMang_dlMang(lM, alpha, fV, pen, w, fM), self.dfMang_dlM(lM, alpha, fV, pen, w, fM), self.dfT_dlT(lT), self.dfT_dlM(lT, lM, pen, w)]
        # vMFunc = lambda dfMpen_dlMpen, dfT_dlT, pen, vMT, lT: self.fiber_velocity(vMT, torch.where(torch.logical_and((dfMpen_dlMpen + dfT_dlT).abs() > eps, lT > self.lT_slack), (dfMpen_dlMpen.div(dfMpen_dlMpen + dfT_dlT)).mul(vMT), vMT), pen).mul(self.vMaxCal(self.vM_max, self.vAlpha, alpha)).div(self.lM_opt*(self.vAlpha + self.vM_max))
        vM_NFunc = lambda dfMpen_dlMpen, dfT_dlT, pen, vMT, lT: self.fiber_velocity(vMT, torch.where(torch.logical_and((dfMpen_dlMpen + dfT_dlT).abs() > eps, lT > self.lT_slack), (dfMpen_dlMpen.div(dfMpen_dlMpen + dfT_dlT)).mul(vMT), vMT), pen).div(self.lM_opt*self.vMaxCal(self.vM_max, self.vAlpha, alpha))
        velocityFunc = lambda alpha, active_force, dfMpen_dlMpen, dfT_dlT, pen, vMT, lT: self.invertForceVelocity(alpha, active_force, vM_NFunc(dfMpen_dlMpen, dfT_dlT, pen, vMT, lT), maxIters=maxIters)

        # initialize params
        lMT = self.lMT(theta, self.lIn, self.lOrig)
        vMT = self.dlMT_dt(theta, omega, self.lIn, self.lOrig)

        lT = 1.01*torch.ones_like(theta)*self.lT_slack
        lM = self.fiberLen(lMT, lT)

        fErr = np.inf*torch.ones_like(theta)

        # calculate the values the first time
        [pen, lT, lM_norm, lT_norm] = positionFunc(lM, lMT)
        [active_force_N, passive_force_N, tendon_force_N] = multipliersFunc(lM_norm, lT_norm)
        velocity_force = torch.ones_like(theta) # initialize this value
        fM = torch.zeros_like(theta) # following the OpenSim code this starts as 0!
        [dfMpen_dlMpen, dfMpen_dlM, dfT_dlT, dfT_dlM] = partialsFunc(alpha, lM, velocity_force, pen, lT, w, fM)
        velocity_force = velocityFunc(alpha, active_force_N, dfMpen_dlMpen, dfT_dlT, pen, vMT, lT)
        fM = fMFunc(alpha, active_force_N, passive_force_N, velocity_force)
        fErr = fErrFunc(alpha, active_force_N, passive_force_N, velocity_force, tendon_force_N, pen)
        [dfMpen_dlMpen, dfMpen_dlM, dfT_dlT, dfT_dlM] = partialsFunc(alpha, lM, velocity_force, pen, lT, w, fM)

        prevErr = fErr
        prevlM = lM

        while (fErr.abs() > tol).any() and iters < maxIters:
            # compute the search direction
            dfErr_dlM = dfMpen_dlM - dfT_dlM
            h = torch.ones_like(theta)

            innerIters = 0
            while (fErr.abs() >= prevErr.abs()).any():
                # take the Newton step
                delta_lM = -h.mul(prevErr.div(dfErr_dlM))

                lM = torch.where(delta_lM.abs() > eps, prevlM + delta_lM, prevlM - delta_lM.sign()*eps.sqrt())
                h = torch.where(delta_lM.abs() > eps, h, 0) # make h go to zero for the ones in the wrong search direction

                lM = self.fiberLenConstrain(lM, self.lM_opt, self.pen_opt)

                [pen, lT, lM_norm, lT_norm] = positionFunc(lM, lMT)
                # print(f'lM: {lM.numpy()} | lM_norm: {lM_norm.numpy()}')
                [active_force_N, passive_force_N, tendon_force_N] = multipliersFunc(lM_norm, lT_norm)
                fM = fMFunc(alpha, active_force_N, passive_force_N, velocity_force)
                fErr = fErrFunc(alpha, active_force_N, passive_force_N, velocity_force, tendon_force_N, pen)

                if (h <= eps.sqrt()).any(): break # unclear if this should be an all() or any() - 07/18: trying any, as we want to break and swap direction if incorrect!
                h *= 0.5
                # h = torch.where(torch.isclose(h, torch.zeros_like(h)), 1, 0.5*h)

                innerIters += 1

            prevErr = fErr
            prevlM = lM

            # fM = fMFunc(alpha, active_force, passive_force, velocity_force)
            [dfMpen_dlMpen, dfMpen_dlM, dfT_dlT, dfT_dlM] = partialsFunc(alpha, lM, velocity_force, pen, lT, w, fM)
            velocity_force = velocityFunc(alpha, active_force_N, dfMpen_dlMpen, dfT_dlT, pen, vMT, lT)

            iters += 1

        lM = self.fiberLenConstrain(lM, self.lM_opt, self.pen_opt)

        if (fErr.abs() < tol).all():
            return lM
        else:
            print(f'Failure fraction: {((fErr.abs() > tol).sum()/torch.numel(fErr)).item()}')
            print(f'Error values: {fErr.numpy()}')
            print(f'Lengths: {lM.numpy()}')
            raise ValueError('Initial muscle length failed to converge')
    
    def invertForceVelocity(self, alpha, active_force_N, vM_N, tol=1e-8, maxIters=10000):
        # for debugging...
        currFrame = inspect.currentframe()
        callFrame = currFrame.f_back
        callFrame = callFrame.f_back # becuase of the lambda function
        callLocals = callFrame.f_locals

        eps = self.EPS.to(device=alpha.device, dtype=alpha.dtype)

        # invert via a loop, rather than explicitly?
        fErr = torch.ones_like(active_force_N)
        fV = torch.ones_like(active_force_N)
        fCE = fV.mul(alpha).mul(active_force_N)

        iters = 0
        while (fErr.abs() >= tol).any() and iters < maxIters:
            vM_N1 = self.i_velocity_force(fCE, alpha.mul(active_force_N))
            fErr = vM_N1 - vM_N

            dvMN_dFm = self.dvM_dFM(alpha.mul(active_force_N), fCE)

            fCE = torch.where(dvMN_dFm.abs() > eps, fCE - fErr.div(dvMN_dFm), fCE)

            if torch.isnan(fErr).any():
                idxs = torch.isnan(fErr).nonzero()
                print(f'Args:\nalpha: {alpha.t().numpy()} | active: {active_force_N.t().numpy()}')# | vM_N: {vM_N.t().numpy()}')
                print(f'Iters: {callLocals["iters"]} | innerIters: {callLocals["innerIters"]} | localIters: {iters}')
                print(f'fV: {fV[idxs].t().numpy()}\nactive: {active_force_N[idxs].t().numpy()}')
                print(f'fCE: {fCE[idxs].t().numpy()}\ndvMN_dFm: {dvMN_dFm[idxs].t().numpy()}\nfErr: {fErr[idxs].t().numpy()}\nvM_N1: {vM_N1[idxs].t().numpy()}\n')

            iters += 1

        # if torch.isclose(active_force_N, torch.zeros_like(active_force_N)).any():
        #     idxs = torch.isclose(active_force_N, torch.zeros_like(active_force_N)).nonzero()

        #     # print(callFrame.f_code.co_name)
        #     assert 'lM_norm' in callLocals

        #     lM_norm = callLocals['lM_norm']
        #     print(f'Iters: {callLocals["iters"]} | innerIters: {callLocals["innerIters"]} | localIters: {iters}')
        #     print(f'iter: {callLocals["iters"]} | lM_norm: {lM_norm[idxs].t().numpy()} | active (passed): {active_force_N[idxs].t().numpy()} | vM: {vM_N[idxs].t().numpy()}')
        
        assert 'lM_norm' in callLocals
        assert 'iters' in callLocals

        lM_norm = callLocals['lM_norm']
        if not callLocals["iters"] % 100: print(f'Iters: {callLocals["iters"]} | innerIters: {callLocals["innerIters"]} | localIters: {iters}'); print(f'iter {callLocals["iters"]}: lM_norm: {lM_norm.t().numpy()} | active (passed): {active_force_N.t().numpy()}')# | vM: {vM_N.t().numpy()}')

        if (fErr.abs() < tol).all():
            return torch.where(fCE.div(alpha.mul(active_force_N)) > 0.0, fCE.div(alpha.mul(active_force_N)), 0.0)
        else:
            # print(fErr)
            print(f'fV Solver Failure Fraction: {torch.logical_or(fErr.abs() >= tol, torch.isnan(fErr)).sum()/torch.numel(fErr)}') # proportion of values that fail to converge
            print(f'Indices: {torch.logical_or(fErr.abs() >= tol, torch.isnan(fErr)).nonzero()}') # indices
            raise ValueError('Solver for force-velocity multiplier failed to converge')
    
############## OLD CODE
    # def bisectionSolve(self, theta, omega, alpha, lowerBounds=0, upperBounds=1, maxIters=10000, tol=1e-4):
    #     batchSize = alpha.shape[0]
    #     device = theta.device

    #     lowerBound = torch.tensor([lowerBounds]*batchSize, dtype=torch.float, device=device)
    #     upperBound = torch.tensor([upperBounds]*batchSize, dtype=torch.float, device=device)
    #     for _ in range(maxIters):
    #         fLeft = self.equilibrateMuscle(lowerBound, theta, omega, alpha)
    #         fRight = self.equilibrateMuscle(upperBound, theta, omega, alpha)

    #         leftSign = fLeft.sign()
    #         rightSign = fRight.sign()

    #         # if torch.any(leftSign == rightSign):
    #         #     raise ValueError(f'Not bounding a force equilibrium with bounds [{lowerBound}, {upperBound}]')
            
    #         mid = (lowerBound + upperBound)/2
    #         fMid = self.equilibrateMuscle(mid, theta, omega, alpha)
    #         midSign = fMid.sign()

    #         if torch.all(fMid.abs() < tol):
    #             return mid

    #         lowerBound = torch.where(leftSign == midSign, mid, lowerBound)
    #         upperBound = torch.where(rightSign == midSign, mid, upperBound)
            
    #     raise ValueError(f'No convergence to valid equilibria within {maxIters} iterations.')
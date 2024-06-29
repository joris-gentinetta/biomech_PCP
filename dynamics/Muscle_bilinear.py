# BUGMAN #1 Jan 29 2022
"""
This class simulates the first order dynamic of one muscle. The dynamic is also compensated by neural network
Muscle dynamic model is from The paper
"Shin D, Kim J, Koike Y (2009) A myokinetic arm model for estimating joint torque and stiffness from EMG signals during maintained posture. Journal of Neurophysiology, 101(1):387–401. https://doi.org/10.1152/jn.00584.2007"
Follings are the values for reference of each parameter modified for lower extrimity from the paper:
K0, N/m 2000
K1, N/m 40000
L0, m   0.03
L1, m   0.006
M,  m   (0.01, 0.05)TA, (-0.03, -0.01)PL    I should use opensim to find better initial value. 
"""

import numpy as np

class Muscle():
    def __init__(self, K0, K1, L0, L1, M, A=-2):        
        """Muscle in the simulation model

        Args:
            K0 (float): K0
            K1 (float): K1
            L0 (float): L0
            L1 (float): L1
            A (float, optional): The nonlinear factor between [-3, 0) to map EMG to muscle activation. Defaults to -2. As A approach to 0, EMG to muscle activation becomes linear.
            M (float): Moment arm of the muscle in the joint of interest.
        """
        
        super().__init__()
        self.K0 = K0
        self.K1 = K1
        self.L0 = L0
        self.L1 = L1
        self.A = A
        self.M = M
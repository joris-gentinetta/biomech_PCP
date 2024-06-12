# Mikey Fernandez 04/02/2022
"""
This class simulates the first order dynamics of one muscle based on the Shadmehr-Arbib 1992 muscle model
"""

import numpy as np
import torch

class Muscle():
    def __init__(self, f_max, lM_opt, k_SE, k_PE, b, gamma):
        
        """Muscle in the simulation model

        Args:
            f_max (float): maximum isometric muscle force, for scaling
            lM_opt (float): optimal muscle fiber length
            k_SE (float): series elastic element constant
            k_PE (float): parallel elastic element constant
            b (float): damping constant
            gamma (float): shape factor for the active force-length curve
        """
        
        super().__init__()
        # define muscle parameters that will be optimized
        self.f_max = f_max
        self.lM_opt = lM_opt
        self.k_SE = k_SE
        self.k_PE = k_PE
        self.b = b
        self.gamma = gamma

    # def muscleState(self, lMT, lM, alpha, f_max, lM_opt, vM_max, lT_slack, device):
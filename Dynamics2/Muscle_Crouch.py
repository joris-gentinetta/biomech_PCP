# Mikey Fernandez 04/02/2022
"""
This class simulates the first order dynamics of one muscle based on the Crouch 2016 Model
"""

class Muscle():
    def __init__(self, f_max, lM_opt, k_PE, v_max, W, c):
        
        """Muscle in the simulation model

        Args:
            f_max (float): maximum isometric muscle force, for scaling
            lM_opt (float): optimal muscle fiber length
            k_PE (float): parallel elastic element constant
            v_max (float): max shortening velocity constant
            W (float): shape factor for the active force-length curve
            c (float): shape factor for the force-velocity curve
        """
        
        super().__init__()
        # define muscle parameters that will be optimized
        self.f_max = f_max
        self.lM_opt = lM_opt
        self.k_PE = k_PE
        self.v_max = v_max
        self.W = W
        self.c = c

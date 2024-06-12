# Mikey Fernandez 07/19/2023
"""
This class defines a single muscle as described by Hogan in "Adaptive Control of Mechanical Impedance by Coactivation of Antagonist Muscles" (1984)
https://summerschool.stiff-project.org/uploads/tx_sibibtex/Hogan_-_Adaptive_control_of_mechanical_impedance_by_coactivation_of_antagonist_muscles.pdf

This muscle model definition allows the individual control of both joint position and joint stiffness
"""

class Muscle():
    def __init__(self, T, K):        
        """Muscle in the simulation model

        Args:
            T (float): maximum muscle isometric torque
            K (float): muscle angular stiffness
        """
        
        super().__init__()
        self.T = T
        self.K = K
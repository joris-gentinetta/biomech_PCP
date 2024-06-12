import torch
import torch.nn as nn

class RKIntegrator(nn.Module):
    def __init__(self, f, order=4):
        """
        Initialize the RK4 integrator.

        Args:
            f: A function representing the system of differential equations dx/dt = f(x, u)
        """
        super(RKIntegrator, self).__init__()
        self.f = f
        self.order = order
        self.setCoefficients()

    def forward(self, SS, Alphas, dt=0.0166666667):
        """
        Perform one step of RK4 integration.

        Args:
            SS: The initial state vector.
            Alphas: The initial inputs
            dt: The timestep
        Returns:
            The updated state vector at time dt.

        """

        k = [torch.zeros_like(SS) for _ in range(self.order)]

        k[0] = dt*self.f(SS, Alphas)
        runSum = torch.zeros_like(SS)
        for i in range(1, self.order):
            tempSS = SS + torch.sum(torch.stack([a * k[j] for j, a in enumerate(self.a[i])]), dim=1)
            k[i] = dt*self.f(tempSS, Alphas)
            runSum += self.b[i]*k[i]

        SS_out = SS + runSum
        return SS_out

        # k1 = dt*self.f(SS, Alphas)
        # k2 = dt*self.f(SS + 0.5*k1, Alphas)
        # k3 = dt*self.f(SS + 0.5*k2, Alphas)
        # k4 = dt*self.f(SS + k3, Alphas)

        # SS_out = SS + (k1 + 2*k2 + 2*k3 + k4)/6.0

        # return SS_out

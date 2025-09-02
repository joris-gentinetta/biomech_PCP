import torch
from matplotlib import pyplot as plt


class ShiftedReLU(torch.nn.Module):
    def forward(self, X):
        return torch.functional.F.relu(X - 0.1)


class Exponential(torch.nn.Module):
    def forward(self, X):
        return torch.exp(X)


class Sigmoid(torch.nn.Module):
    def forward(self, X):
        return torch.sigmoid(X)


if __name__ == "__main__":
    shifted_relu = ShiftedReLU()
    exponential = Exponential()
    plt.plot(torch.linspace(-1, 1, 1000), shifted_relu(torch.linspace(-1, 1, 1000)))
    plt.plot(torch.linspace(-1, 1, 1000), exponential(torch.linspace(-1, 1, 1000)))
    plt.show()

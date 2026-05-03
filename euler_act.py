import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveEulerActivation(nn.Module):
    """
    Implementation of Euler Matrix Activation (EMA).
    Reference: Mantousis, V. (2026). Adaptive Euler Matrix Activation.
    """
    def __init__(self, alpha=1.0, theta=0.5):
        super(AdaptiveEulerActivation, self).__init__()
        # Learnable parameters
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.theta = nn.Parameter(torch.tensor([theta]))

    def forward(self, x):
        # f(x) = alpha * (x * cos(theta) - tanh(x) * sin(theta))
        return self.alpha * (x * torch.cos(self.theta) - torch.tanh(x) * torch.sin(self.theta))
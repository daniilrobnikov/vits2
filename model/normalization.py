import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class CondLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5, cond_channels=0):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.linear_gamma = nn.Linear(cond_channels, channels)
        self.linear_beta = nn.Linear(cond_channels, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        gamma = self.linear_gamma(cond)
        beta = self.linear_beta(cond)

        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), gamma, beta, self.eps)
        return x.transpose(1, -1)

import torch
import torch.nn as nn

from model.transformer import RelativePositionTransformer
from model.modules import WN, Flip


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0, use_transformer_flow=True):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            use_transformer = use_transformer_flow if i == n_flows - 1 else False
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True, use_transformer_flow=use_transformer))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


# TODO rewrite for 256x256 attention score map
# TODO Test another architecture for Transformer block
# TODO (RelativePositionTransformer -> Conv1d -> WN -> Conv1d) -> Flip
# TODO RelativePositionTransformer -> Flip -> (Conv1d -> WN -> Conv1d) -> Flip
class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False, use_transformer_flow=True):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre_transformer = (
            RelativePositionTransformer(
                self.half_channels,
                self.half_channels,
                self.half_channels,
                self.half_channels,
                n_heads=2,
                n_layers=1,
                kernel_size=3,
                dropout=0.1,
                window_size=None,
            )
            if use_transformer_flow
            else None
        )

        self.pre = nn.Linear(self.half_channels, hidden_channels)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Linear(hidden_channels, self.half_channels * (2 - mean_only))
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        x0_ = x0
        if self.pre_transformer is not None:
            x0_ = self.pre_transformer(x0 * x_mask, x_mask)
            x0_ = x0_ + x0  # residual connection
        h = self.pre(x0_.mT).mT * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h.mT).mT * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

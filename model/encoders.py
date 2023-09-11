import math
import torch
import torch.nn as nn

from model.modules import WN
from model.transformer import RelativePositionTransformer
from utils.model import sequence_mask


# * Ready and Tested
class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        dropout: float,
        gin_channels=0,
        lang_channels=0,
        speaker_cond_layer=0,
    ):
        """Text Encoder for VITS model.

        Args:
            n_vocab (int): Number of characters for the embedding layer.
            out_channels (int): Number of channels for the output.
            hidden_channels (int): Number of channels for the hidden layers.
            hidden_channels_ffn (int): Number of channels for the convolutional layers.
            n_heads (int): Number of attention heads for the Transformer layers.
            n_layers (int): Number of Transformer layers.
            kernel_size (int): Kernel size for the FFN layers in Transformer network.
            dropout (float): Dropout rate for the Transformer layers.
            gin_channels (int, optional): Number of channels for speaker embedding. Defaults to 0.
            lang_channels (int, optional): Number of channels for language embedding. Defaults to 0.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            window_size=4,
            gin_channels=gin_channels,
            lang_channels=lang_channels,
            speaker_cond_layer=speaker_cond_layer,
        )
        self.proj = nn.Linear(hidden_channels, out_channels * 2)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, g: torch.Tensor = None, lang: torch.Tensor = None):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """
        x = self.emb(x).mT * math.sqrt(self.hidden_channels)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x, x_mask, g=g, lang=lang)
        stats = self.proj(x.mT).mT * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs) * x_mask
        return z, m, logs, x, x_mask


# * Ready and Tested
class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels=0,
    ):
        """Posterior Encoder of VITS model.

        ::
            x -> conv1x1() -> WaveNet() (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z

        Args:
            in_channels (int): Number of input tensor channels.
            out_channels (int): Number of output tensor channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of the WaveNet convolution layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            cond_channels (int, optional): Number of conditioning tensor channels. Defaults to 0.
        """
        super().__init__()
        self.out_channels = out_channels

        self.pre = nn.Linear(in_channels, hidden_channels)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Linear(hidden_channels, out_channels * 2)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, g=None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
            - g: :math:`[B, C, 1]`
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x.mT).mT * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x.mT).mT * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs) * x_mask
        return z, m, logs, x_mask


# TODO Ready for testing
class AudioEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        dropout: float,
        gin_channels=0,
        lang_channels=0,
        speaker_cond_layer=0,
    ):
        """Audio Encoder of VITS model.

        Args:
            in_channels (int): Number of input tensor channels.
            out_channels (int): Number of channels for the output.
            hidden_channels (int): Number of channels for the hidden layers.
            hidden_channels_ffn (int): Number of channels for the convolutional layers.
            n_heads (int): Number of attention heads for the Transformer layers.
            n_layers (int): Number of Transformer layers.
            kernel_size (int): Kernel size for the FFN layers in Transformer network.
            dropout (float): Dropout rate for the Transformer layers.
            gin_channels (int, optional): Number of channels for speaker embedding. Defaults to 0.
            lang_channels (int, optional): Number of channels for language embedding. Defaults to 0.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.pre = nn.Linear(in_channels, hidden_channels)
        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            window_size=4,
            gin_channels=gin_channels,
            lang_channels=lang_channels,
            speaker_cond_layer=speaker_cond_layer,
        )
        self.post = nn.Linear(hidden_channels, out_channels * 2)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, g: torch.Tensor = None, lang: torch.Tensor = None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
            - g: :math:`[B, C, 1]`
            - lang: :math:`[B, C, 1]`
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.pre(x.mT).mT * x_mask  # [B, C, t']
        x = self.encoder(x, x_mask, g=g, lang=lang)
        stats = self.post(x.mT).mT * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs) * x_mask
        return z, m, logs, x_mask

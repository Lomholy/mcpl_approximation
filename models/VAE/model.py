import torch
from torch.nn import functional as F
from torch import nn as nn

class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x):
        y = self.norm(x)
        y = self.fc2(F.silu(self.fc1(y)))
        return x + y


class VAE(nn.Module):
    def __init__(self, input_dim=12, latent_dim=10, width=256, dec_width=256, depth=5):
        super().__init__()
        self.lat_dim = latent_dim

        # Encoder
        self.in_proj = nn.Linear(input_dim, width)
        self.enc_blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])

        self.fc_mu  = nn.Linear(width, latent_dim)
        self.fc_sig = nn.Linear(width, latent_dim)

        # Decoder
        self.dec_in = nn.Linear(latent_dim, width)
        self.dec_blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.dec_out = nn.Linear(width, input_dim)

    def encode(self, x):
        x = self.in_proj(x)
        for block in self.enc_blocks:
            x = block(x)
        logvar = self.fc_sig(x)
        logvar = torch.clamp(logvar, -20, 10)

        return self.fc_mu(x), logvar

    def decode(self, z):
        z = self.dec_in(z)
        for block in self.dec_blocks:
            z = block(z)
        return self.dec_out(z)

    def forward(self, x):
        mu, sig = self.encode(x)
        std = torch.exp(0.5 * sig).clamp(min=1e-6)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, sig


class Sampler(nn.Module):
    """Generative wrapper exposing a single Gaussian-noise-in / neutron-phase-space-out
    call, matching the interface Source_ML(_torch).comp expects (and that CFM's own
    Sampler in ../CFM/model.py provides): the mcstas components always fill a
    12-wide Gaussian noise buffer, but the VAE's decoder only takes `lat_dim` (10)
    latent inputs, so only the first `lat_dim` columns of the noise buffer are used.
    """

    def __init__(self, vae: VAE, device="cpu"):
        super().__init__()
        self.vae = vae.to(device)
        self.lat_dim = vae.lat_dim

    def forward(self, x):
        z = x[:, : self.lat_dim]
        return self.vae.decode(z)


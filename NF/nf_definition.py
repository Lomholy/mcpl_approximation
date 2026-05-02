import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Define the model
class FourierTime(nn.Module):
    def __init__(self, dim=128, max_freq=1000.0):
        super().__init__()

        half = (dim - 1) // 2
        freqs = torch.exp(torch.linspace(0, math.log(max_freq), half))

        self.register_buffer("freqs", freqs)
        self.out_dim = 1 + 2 * half

    def forward(self, t):
        if t.ndim == 2:
            t = t[:, 0]

        ang = 2.0 * math.pi * t[:, None] * self.freqs[None, :]
        return torch.cat([t[:, None], torch.sin(ang), torch.cos(ang)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, width, emb_dim):
        super().__init__()

        self.norm = nn.LayerNorm(width)
        self.emb = nn.Linear(emb_dim, 2 * width)

        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x, e):
        scale, shift = self.emb(e).chunk(2, dim=-1)

        y = self.norm(x)
        y = y * (1.0 + scale) + shift
        y = self.fc2(F.silu(self.fc1(y)))

        return x + y


class VelocityField(nn.Module):
    def __init__(
        self,
        input_dim=7,
        width=512,
        depth=8,
        time_dim=128,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.time = FourierTime(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time.out_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

        self.in_proj = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([ResBlock(width, width) for _ in range(depth)])

        self.out = nn.Sequential(
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, input_dim),
        )

        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        e = self.time_mlp(self.time(t))

        x = self.in_proj(x) + e

        for block in self.blocks:
            x = block(x, e)

        return self.out(x)

    def time_embed(self, t, max_positions=10000):
        # Embed t into a trigonometric space. Half of inputs are cosine, other half are sine
        t = t.squeeze(-1) * max_positions
        half_dim = self.t_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.t_dim % 2 == 1:  # zero padding
            emb = nn.functional.pad(emb, (0, 1), mode="constant")
        return emb


def sample_t(batch_size, device):
    mode = torch.rand(batch_size, 1, device=device)

    t_uniform = torch.rand(batch_size, 1, device=device)
    t_near_1 = 1.0 - torch.rand(batch_size, 1, device=device).pow(2)
    t_near_0 = torch.rand(batch_size, 1, device=device).pow(2)

    t = torch.where(
        mode < 0.70,
        t_uniform,
        torch.where(mode < 0.90, t_near_1, t_near_0),
    )

    return t

import torch.nn as nn
import torch
import torch.nn.functional as F


class Linear(nn.Linear):
    """nn.Linear computed as x @ W.T + b. torch.nn.functional.linear with a bias
    returns wrong values on MPS in some PyTorch builds (verified on 2.13.0),
    while the matmul form is exact. The parameters and state_dict keys are
    those of nn.Linear."""

    def forward(self, x):
        return x @ self.weight.t() + self.bias



# Define the model
class ResBlock(nn.Module):
    def __init__(self, width, emb_dim):
        super().__init__()

        self.norm = nn.Identity(width)
        self.emb = Linear(emb_dim, 2 * width)

        self.fc1 = Linear(width, 4 * width)
        self.fc2 = Linear(4 * width, width)

    def forward(self, x, e):
        scale, shift = self.emb(e).chunk(2, dim=-1)

        y = self.norm(x)
        y = y * (1.0 + scale) + shift
        y = self.fc2(F.silu(self.fc1(y)))

        return x + y


class VelocityField(nn.Module):
    def __init__(
        self,
        input_dim=12,
        width=64,
        depth=3,
        time_dim=28,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.time_mlp = nn.Sequential(
            Linear(1, 64),
            nn.SiLU(),
            Linear(64, 64),
            nn.SiLU(),
            Linear(64, width),
        )

        self.in_proj = Linear(input_dim, width)
        self.blocks = nn.ModuleList([ResBlock(width, width) for _ in range(depth)])

        self.out = nn.Sequential(
            nn.Identity(width),
            nn.SiLU(),
            Linear(width, input_dim),
        )

        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        e = self.time_mlp(t)

        x = self.in_proj(x) + e

        for block in self.blocks:
            x = block(x, e)

        return self.out(x)


class Sampler(nn.Module):
    def __init__(self, velocity_model, n_steps=64, device="cpu"):
        super().__init__()

        self.velocity = velocity_model.to(device)
        self.n_steps = n_steps
        self.register_buffer(
            "time_table", torch.linspace(0, 1, n_steps, device=device).view(n_steps, 1, 1)
        )

    def forward(self, x):

        dt = 1.0 / self.n_steps
        for step in range(self.n_steps):
            t = self.time_table[step]
            x1 = self.velocity(x, t)
            x = x + dt * x1
        return x

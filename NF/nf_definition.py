import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------------------------------------
#  Neural network for the velocity field v_theta(x, t)
# ---------------------------------------------------------


class VelocityField(nn.Module):
    def __init__(self, input_dim=6, start = 12, mid = 512, network_dimensions=512):
        super().__init__()
        self.t_dim = input_dim
        self.apply_model = nn.Sequential(
                nn.Linear(start, start * 2),
                nn.LeakyReLU(),
                nn.Linear(start * 2, start * 2**2),
                nn.LeakyReLU(),
                nn.Linear(start * 2**2, start * 2**3),
                nn.LeakyReLU(),
                nn.Linear(start * 2**3, start * 2**4),
                nn.LeakyReLU(),
                nn.Linear(start * 2**4, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Linear(512, start * 2**4),
                nn.LeakyReLU(),
                nn.Linear(start * 2**4, start * 2**3),
                nn.LeakyReLU(),
                nn.Linear(start * 2**3, start * 2**2),
                nn.LeakyReLU(),
                nn.Linear(start * 2**2, start * 2),
                nn.LeakyReLU(),
                nn.Linear(start * 2, input_dim)
        )

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

    def forward(self, x, t):
        # Embed time t (shape: [batch_size, 1]) into a higher-dimensional vector
        t_embed = self.time_embed(t)
        x = torch.column_stack((x, t_embed))
        x = self.apply_model(x)
        return x

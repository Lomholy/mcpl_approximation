import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------------------------------------
#  Neural network for the velocity field v_theta(x, t)
# ---------------------------------------------------------


class Block(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.ff = nn.Linear(channels, channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ff(x))

class VelocityField(nn.Module):
    def __init__(self,
                 input_dim=5,
                 layers=5,
                 time_dimensions=512,
                 network_dimensions=512
                 ):
        super().__init__()
        self.t_dim = time_dimensions
        self.t_in = nn.Linear(time_dimensions, network_dimensions)
        self.x_in = nn.Linear(input_dim, network_dimensions)
        self.blocks = nn.Sequential(*[
            Block(network_dimensions) for _ in range(layers)
            ])
        self.net = nn.Sequential(nn.Linear(network_dimensions, input_dim))

    def time_embed(self, t, max_positions=10000):
        # Embed t into a trigonometric space. Half of inputs are cosine, other half are sine 
        t = t.squeeze(-1) * max_positions
        half_dim = self.t_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.t_dim % 2 == 1:  # zero padding
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def forward(self, x, t):
        # Embed time t (shape: [batch_size, 1]) into a higher-dimensional vector
        t_embed = self.time_embed(t)
        t_embed = self.t_in(t_embed)

        x = self.x_in(x)
        x = x + t_embed
        x = self.blocks(x)
        x = self.net(x)

        # Pass through the network to predict the velocity at (x, t)
        return x

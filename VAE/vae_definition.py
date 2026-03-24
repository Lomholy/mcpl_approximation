import torch
from torch.nn import functional as F
from torch import nn as nn


def vae_loss(recon_x, x, mu, logvar, kl_weight):
    # Latent (KL divergence) loss
    latent_loss = 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1 - logvar)

    # Reconstruction loss (L1)
    reconstruction_loss = torch.mean(torch.abs(x - recon_x))

    # Total VAE loss
    vae_loss = reconstruction_loss + kl_weight * latent_loss

    return vae_loss, reconstruction_loss, latent_loss


class VAE(nn.Module):
    def __init__(self, input_dim=7, latent_dim=1028, out_act="sigmoid"):

        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ---- Encoder ----

        self.enc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.PReLU(),
            nn.Linear(64, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, latent_dim),
            nn.PReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        # ---- Decoder ----

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )
        self.out_act = {
            "linear": nn.Identity(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }[out_act]

    def encode(self, x):
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # z

    def decode(self, z):
        h = self.dec(z)
        return self.out_act(h)  # no activation → regression

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

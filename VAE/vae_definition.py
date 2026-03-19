import torch
from torch.nn import functional as F
from torch import nn as nn


def vae_loss(recon_x, x, mu, logvar):
    recon = F.mse_loss(recon_x, x, reduction="mean") * x.shape[1]  # scale to dim
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    loss = recon + kl

    return loss


def gaussian_nll(mu_x, logvar_x, x):
    return 0.5 * torch.sum(
        logvar_x
        + (x - mu_x) ** 2 / torch.exp(logvar_x)
        + torch.log(torch.tensor(2 * 3.141592653589793))
    )


def vae_loss_gauss(mu_x, logvar_x, x, mu, logvar):
    recon = gaussian_nll(mu_x, logvar_x, x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl


class VAE(nn.Module):
    def __init__(self, input_dim=8, latent_dim=32, out_act="sigmoid"):
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
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # ---- Decoder ----

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
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

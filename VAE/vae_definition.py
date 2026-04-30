import torch
from torch.nn import functional as F
from torch import nn as nn
import numpy as np


def sigmoid(x, top, center, slope=0.03):
    return (top)/(1+np.exp((center - x)*slope))


def vae_loss(recon_x, x, mu, sig, kl_weight, current_epoch, total_epochs):
    # Latent (KL divergence) loss
    var = sig**2
    if current_epoch > 1/3 * total_epochs:
        latent_loss = 0.5 * torch.sum(-1 - torch.log(var) + var + mu**2)
        latent_loss *= sigmoid(current_epoch, top=kl_weight, center=1/2*total_epochs)
    else:
        latent_loss = torch.tensor(0)
    # Reconstruction loss (L1)
    reconstruction_loss = F.mse_loss(recon_x, x, reduction="sum")
    if reconstruction_loss.isnan() or latent_loss.isnan():
        print(f"Nan present in loss!, rec loss {reconstruction_loss.isnan()}, lat loss{latent_loss.isnan()}")
        exit(0)
    # Total VAE loss
    vae_loss = reconstruction_loss + latent_loss
    return vae_loss, reconstruction_loss, latent_loss


class VAE(nn.Module):
    def __init__(self, input_dim=6, latent_dim=256):

        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ---- Encoder ----


        self.enc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, latent_dim)
        )

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_sig = nn.Linear(latent_dim, latent_dim)

        # ---- Decoder ----

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, input_dim)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)



    def encode(self, x):
        h = self.enc(x)
        mu = self.fc_mu(h)
        sig = F.softplus(self.fc_sig(h)) + 1e-6
        
        if mu.isnan().any() or sig.isnan().any():
            print(f"NaN in encoder output {x}")
            print(f"mu={mu}")
            print(f"sig={sig}")

            exit()
        return mu, sig

    def reparameterize(self, mu, sig):
        std = sig
        eps = torch.randn_like(std)
        return mu + eps * std  # z

    def decode(self, z):
        h = self.dec(z)
        return h

    def forward(self, x):
        mu, sig = self.encode(x)
        z = self.reparameterize(mu, sig)
        recon = self.decode(z)
        return recon, mu, sig

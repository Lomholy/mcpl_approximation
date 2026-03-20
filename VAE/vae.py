import numpy as np
import torch
import os
import mcpl
from torch.utils.data import TensorDataset, DataLoader, random_split
from vae_definition import VAE, vae_loss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "mps"
batch_size = 2048
epochs = 30
patience = 5

vae = VAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), weight_decay=0, lr=1e-3)

mcplfile = mcpl.MCPLFile("../ODIN.mcpl.gz")
n_blocks = 100
data = torch.zeros([n_blocks * 10000, 7], dtype=torch.float32)

for i, p in enumerate(mcplfile.particle_blocks):
    data[i * 10000 : (i + 1) * 10000, 0] = torch.asarray(p.weight)
    data[i * 10000 : (i + 1) * 10000, 1] = torch.asarray(p.ekin)
    data[i * 10000 : (i + 1) * 10000, 2:5] = torch.asarray(
        np.array([p.ux, p.uy, p.uz])
    ).T
    data[i * 10000 : (i + 1) * 10000, 5:] = torch.asarray(np.array([p.x, p.y])).T
    if i == n_blocks - 1:
        break

eps = 1e-9
mins = data.min(0).values
maxs = data.max(0).values
scales = (maxs - mins).clamp_min(eps)
data = ((data - mins) / scales).clamp(0, 1)

dataset = TensorDataset(data)
n_total = len(dataset)
n_val = int(0.1 * n_total)
n_train = n_total - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

best_val = float("inf")
no_improve = 0
losses = []
for epoch in range(epochs):
    kl_weight = 0.05
    vae.train()
    for (x,) in train_loader:
        x = x.to(device)
        recon, mu, logvar = vae(x)
        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, kl_weight=kl_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    vae.eval()
    val_losses = []
    with torch.no_grad():
        for (xv,) in val_loader:
            xv = xv.to(device)
            recon, mu, logvar = vae(xv)
            vloss, _, _ = vae_loss(recon, xv, mu, logvar, kl_weight=kl_weight)
            val_losses.append(vloss.item())
    val_mean = np.mean(val_losses)

    print(f"Epoch {epoch}  train={loss.item():.4g}  val={val_mean:.4g}")

    if val_mean < best_val:
        best_val = val_mean
        no_improve = 0
        torch.save(
            {
                "state_dict": vae.state_dict(),
                "mins": mins,
                "maxs": maxs,
                "scales": scales,
            },
            "vae_best.pth",
        )
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

np.save("vae_losses.npy", losses)

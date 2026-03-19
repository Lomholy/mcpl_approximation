import numpy as np
import torch
import os
import mcpl
from torch.utils.data import TensorDataset, DataLoader
from vae_definition import VAE, vae_loss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "mps"
batch_size = 2048
epochs = 5

vae = VAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), weight_decay=1e-2, lr=1e-4)

mcplfile = mcpl.MCPLFile("./ODIN.mcpl.gz")
n_blocks = 100
data = torch.zeros([n_blocks * 10000, 8], dtype=torch.float32)

for i, p in enumerate(mcplfile.particle_blocks):
    data[i * 10000 : (i + 1) * 10000, 0] = torch.asarray(p.weight)
    data[i * 10000 : (i + 1) * 10000, 1] = torch.asarray(p.ekin)
    data[i * 10000 : (i + 1) * 10000, 2:5] = torch.asarray(
        np.array([p.ux, p.uy, p.uz])
    ).T
    data[i * 10000 : (i + 1) * 10000, 5:] = torch.asarray(np.array([p.x, p.y, p.z])).T
    if i == n_blocks - 1:
        break

eps = 1e-9
mins = data.min(0).values
maxs = data.max(0).values
scales = (maxs - mins).clamp_min(eps)
data = ((data - mins) / scales).clamp(0, 1)

loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

losses = []

for epoch in range(epochs):
    for (x,) in loader:
        x = x.to(device)
        recon, mu, logvar = vae(x)
        loss = vae_loss(recon, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    if epoch % 1 == 0:
        print(epoch, loss.item())

torch.save(
    {
        "state_dict": vae.state_dict(),
        "mins": mins,
        "maxs": maxs,
        "scales": scales,
    },
    "vae.pth",
)

np.save("vae_losses.npy", losses)

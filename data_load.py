# Author: Daniel Lomholt Christensen
import mcpl
import numpy as np
import torch


def load_mcpl_file(filepath, n_blocks):
    mcplfile = mcpl.MCPLFile(filepath)
    data = np.zeros([n_blocks * 10000, 7], dtype=np.float32)

    for i, p in enumerate(mcplfile.particle_blocks):
        data[i * 10000 : (i + 1) * 10000, 0] = np.asarray(p.weight)
        data[i * 10000 : (i + 1) * 10000, 1] = np.asarray(p.ekin)
        data[i * 10000 : (i + 1) * 10000, 2:5] = np.asarray([p.ux, p.uy, p.uz]).T
        data[i * 10000 : (i + 1) * 10000, 5:] = np.asarray([p.x, p.y]).T
        if i == n_blocks - 1:
            break
    return torch.tensor(data, dtype=torch.float32)


def dim_reduction(data):
    #
    out = torch.zeros((data.shape[0], 6))
    out[:, 0:2] = data[:, 0:2]
    ux, uy, uz = data[:, 2], data[:, 3], data[:, 4]
    theta = torch.atan2(torch.sqrt(ux * ux + uy * uy), uz)
    phi = torch.atan2(uy, ux)
    out[:, 2] = theta
    out[:, 3] = phi
    out[:, 4:] = data[:, 5:]
    return out


def preprocess(data):
    data = dim_reduction(data)
    eps = 1e-12
    data[:, 0] = torch.log(data[:, 0] + eps)  # weight
    data[:, 1] = torch.log(data[:, 1] + eps)  # energy

    # Make sure angles are bounded between + pi and +- pi
    theta = data[:, 2]
    phi = data[:, 3]
    theta = torch.clamp(theta / torch.pi, eps, 1 - eps)
    phi = torch.clamp((phi + torch.pi) / 2 / torch.pi, eps, 1 - eps)
    theta = torch.logit(theta, eps=eps)
    phi = torch.logit(phi, eps=eps)

    data[:, 2] = theta
    data[:, 3] = phi

    # For positions x,y:
    xmin = data[:, 4].min()
    xmax = data[:, 4].max()
    ymin = data[:, 5].min()
    ymax = data[:, 5].max()
    dx = xmax - xmin
    dy = ymax - ymin
    # Use lower eps for clamps
    eps = 1e-7
    x = torch.clamp((data[:, 4] - xmin) / dx, eps, 1 - eps)
    y = torch.clamp((data[:, 5] - ymin) / dy, eps, 1 - eps)

    data[:, 4] = torch.logit(x, eps=eps)
    data[:, 5] = torch.logit(y, eps=eps)

    return data, xmin, ymin, dx, dy


def normalize(data):
    # Normalizes between 0 and 1
    xmin = data.min()
    xmax = data.max()
    dx = xmax - xmin
    data = (data - xmin) / dx
    return data, xmin, dx


def preprocess_nn(data):
    data = dim_reduction(data)
    mins = np.zeros((data.shape[1]))
    dxs = np.zeros((data.shape[1]))
    for i in range(data.shape[1]):
        data[:, i], mins[i], dxs[i] = normalize(data[:, i])

    return data, mins, dxs


def postprocess(data, xmin, ymin, dx, dy):
    data[:, 0] = torch.exp(data[:, 0])
    data[:, 1] = torch.exp(data[:, 1])
    theta = torch.special.expit(data[:, 2])
    theta *= torch.pi
    phi = torch.special.expit(data[:, 3])
    phi *= 2 * torch.pi
    phi -= torch.pi
    data[:, 2] = theta
    data[:, 3] = phi

    x = torch.special.expit(data[:, 4])
    y = torch.special.expit(data[:, 5])
    x *= dx
    y *= dy
    x += xmin
    y += ymin

    data[:, 4] = x
    data[:, 5] = y

    return data


def inverse_norm(data, min, dx):
    data = data * dx + min
    return data


def postprocess_nn(data, mins, dxs, filename=""):
    if filename != "":
        mins, dxs = np.load(filename)
    for i in range(data.shape[1]):
        data[:, i] = inverse_norm(data[:, i], mins[i], dxs[i])
    return data

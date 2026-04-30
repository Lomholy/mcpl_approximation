# Author: Daniel Lomholt Christensen
import mcpl
import numpy as np
import torch
import np2mcpl


def load_mcpl_file(
    filepath: str, n_particles: int, offset: int = 0, print_status: bool = False
):
    mcplfile = mcpl.MCPLFile(filepath)
    data = np.zeros([n_particles, 7], dtype=np.float32)

    for i, p in enumerate(mcplfile.particles):
        if i < offset:
            continue
        if i % 10000 == 0 and print_status == True:
            print(f"Loading particles. {i - offset} currently loaded.")
        j = i - offset
        data[j : (j + 1), 0] = np.asarray(p.weight)
        data[j : (j + 1), 1] = np.asarray(p.ekin)
        data[j : (j + 1), 2:5] = np.asarray([p.ux, p.uy, p.uz]).T
        data[j : (j + 1), 5:] = np.asarray([p.x, p.y]).T
        if j >= n_particles + offset:
            break
    return torch.tensor(data, dtype=torch.float32)


def dim_reduction(data):
    out = torch.zeros((data.shape[0], 6))
    out[:, 0:2] = data[:, 0:2]
    ux, uy, uz = data[:, 2], data[:, 3], data[:, 4]
    theta = torch.atan2(torch.sqrt(ux * ux + uy * uy), uz)
    phi = torch.atan2(uy, ux)
    out[:, 2] = theta
    out[:, 3] = phi
    out[:, 4:] = data[:, 5:7]
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
    data[:, 0] = torch.log(data[:, 0]*1e20 + 1)  # weight
    data[:, 1] = torch.log(data[:, 1]*1e10 + 1)  # energy
    for i in range(data.shape[1]):
        data[:, i], mins[i], dxs[i] = normalize(data[:, i])
    return data, mins, dxs


def postprocess(data, xmin, ymin, dx, dy):
    output = torch.zeros((data.shape[0], 7))
    output[:, 0] = torch.exp(data[:, 0])
    output[:, 1] = torch.exp(data[:, 1])
    theta = torch.special.expit(data[:, 2])
    theta *= torch.pi
    phi = torch.special.expit(data[:, 3])
    phi *= 2 * torch.pi
    phi -= torch.pi
    data[:, 2] = theta
    data[:, 3] = phi
    output[:, 2] = np.cos(data[:, 3]) * np.sin(data[:, 2])
    output[:, 3] = np.sin(data[:, 3]) * np.sin(data[:, 2])
    output[:, 4] = np.cos(data[:, 2])

    x = torch.special.expit(data[:, 4])
    y = torch.special.expit(data[:, 5])
    x *= dx
    y *= dy
    x += xmin
    y += ymin

    output[:, 5] = x
    output[:, 6] = y
    return output


def inverse_norm(data, min, dx):
    data = data * dx + min
    return data


def postprocess_nn(data, mins=None, dxs=None, filename=""):
    if filename != "":
        mins, dxs = np.load(filename)
    for i in range(data.shape[1]):
        data[:, i] = inverse_norm(data[:, i], mins[i], dxs[i])
    output = torch.zeros((data.shape[0], 7))
    output[:, 0] = (np.exp(data[:, 0]) - 1)/1e20
    output[:, 1] = (np.exp(data[:, 1]) - 1)/1e10

    output[:, 2] = np.cos(data[:, 3]) * np.sin(data[:, 2])
    output[:, 3] = np.sin(data[:, 3]) * np.sin(data[:, 2])
    output[:, 4] = np.cos(data[:, 2])
    output[:, 5:] = data[:, 4:]

    return output


def save_data_as_mcpl(data, filename):
    # Convert the input data which is 6d, to np2mcpl data, which is 10 d
    output = np.zeros((data.shape[0], 10))
    output[:, 0] = 2112  # PDG code
    output[:, 1] = data[:, 5]  # Position in xy
    output[:, 2] = data[:, 6]  # Position in xy
    output[:, 3] = 6000  # Position in Z
    # Convert angles into direction vector
    output[:, 4] = data[:, 2]
    output[:, 5] = data[:, 3]
    output[:, 6] = data[:, 4]

    output[:, 7] = 1 # Time is irrelevant here
    output[:, 8] = data[:, 1] # Energy
    output[:, 9] = data[:, 0] # Weight

    np2mcpl.save(filename, output)

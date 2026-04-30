import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_correlations_7d(data, title="", filename=""):
    labels = [
        "Weight [a.u]",
        "Energy [MeV]",
        "Direction x",
        "Direction y",
        "Direction z - 1",
        "Position x [cm]",
        "Position y [cm]",
    ]
    # labels = ["Angle theta", "Angle phi", "Position x", "Position y"]
    input_dim = 7
    
    plot_data = torch.asarray(data, copy=True)
    print(plot_data[0])
    plot_data[:,4] -= 1
    print(plot_data[0])
    fig, ax = plt.subplots(ncols=input_dim, nrows=input_dim, figsize=(20,20))
    for i in range(input_dim):
        for j in range(input_dim):
            if j > i:
                ax[i, j].set_axis_off()
                continue
            if j == i:
                try:
                    ax[i, j].hist(plot_data[:, i], bins=50)
                    ax[i, j].set(xlabel=labels[i], ylabel="Counts")
                except Exception as e:
                    print(e)
                    ax[i, j].set_axis_off()
                continue
            ax[i, j].hist2d(plot_data[:, i], plot_data[:, j], bins=100)
            ax[i, j].set(xlabel=labels[i], ylabel=labels[j])
    fig.suptitle(title)
    fig.tight_layout()
    if filename != "":
        fig.savefig(filename, dpi=300)


def plot_correlations_6d(data, title="", filename=""):
    labels = [
        "Weight [a.u]",
        "Energy [AA]",
        "Theta [rad]",
        "Phi [rad]",
        "Position x [cm]",
        "Position y [cm]",
    ]
    input_dim = 6
    fig, ax = plt.subplots(ncols=input_dim, nrows=input_dim, figsize=(13, 13))
    for i in range(input_dim):
        for j in range(input_dim):
            if j > i:
                ax[i, j].set_axis_off()
                continue
            if j == i:
                try:
                    ax[i, j].hist(data[:, i], bins=50)
                    ax[i, j].set(xlabel=labels[i], ylabel="Counts [#]")
                except Exception as e:
                    print(e)
                    ax[i, j].set_axis_off()
                continue

            ax[i, j].hist2d(data[:, i], data[:, j], bins=100)
            ax[i, j].set(xlabel=labels[i], ylabel=labels[j])

    fig.suptitle(title)
    fig.tight_layout()
    if filename !="":
        fig.savefig(filename, dpi=300)


def plot_correlations_5d(data, title="", weights = np.array([0])):
    labels = [
        "Energy",
        "Theta",
        "Phi",
        "Position x",
        "Position y",
    ]
    input_dim = 5
    fig, ax = plt.subplots(ncols=input_dim, nrows=input_dim, figsize=(15, 20))
    for i in range(input_dim):
        for j in range(input_dim):
            if j > i:
                ax[i, j].set_axis_off()
                continue
            if j == i:
                try:
                    ax[i, j].hist(data[:, i], bins=50)
                    ax[i, j].set(xlabel=labels[i], ylabel="Counts")
                except Exception as e:
                    print(e)
                    ax[i, j].set_axis_off()
                continue
            if weights[0] == 0:
                ax[i, j].hist2d(data[:, i], data[:, j], bins=100)
            else:
                ax[i, j].hist2d(data[:, i], data[:, j], weights=weights, bins=100)

            ax[i, j].set(xlabel=labels[i], ylabel=labels[j])

    fig.suptitle(title)
    fig.tight_layout()

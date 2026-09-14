import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable


def plot_correlations_12d(data, title="", filename=""):
    labels = [
        "Weight [a.u]",
        "Energy [MeV]",
        "Time [ms]",
        "Direction x",
        "Direction y",
        "Direction z",
        "Position x [cm]",
        "Position y [cm]",
        "Position z [cm]",
        "Polarization x",
        "Polarization y",
        "Polarization z",
    ]
    # labels = ["Angle theta", "Angle phi", "Position x", "Position y"]
    input_dim = 12

    plot_data = torch.asarray(data, copy=True)

    # --- compute global histogram range ---
    bins = 250
    h_all = []

    for i in range(input_dim):
        for j in range(i):
            h, _, _ = np.histogram2d(plot_data[:, i], plot_data[:, j], bins=bins)
            h_all.append(h.ravel())

    h_all = np.concatenate(h_all)
    h_all = h_all[h_all > 0]  # avoid log(0)

    norm = LogNorm(vmin=h_all.min(), vmax=h_all.max())

    fig, ax = plt.subplots(ncols=input_dim, nrows=input_dim, figsize=(20, 20))
    for i in range(input_dim):
        for j in range(input_dim):
            if j > i:
                ax[i, j].set_axis_off()
                continue
            if j == i:
                try:
                    ax[i, j].hist(plot_data[:, i], bins=100)
                    ax[i, j].set(xlabel=labels[i], ylabel="Counts")
                except Exception:
                    try:
                        ax[i, j].hist(plot_data[:, i], bins=50)
                        ax[i, j].set(xlabel=labels[i], ylabel="Counts")
                    except Exception as e:
                        print(e)
                        ax[i, j].set_axis_off()
                continue
            ax[i, j].hist2d(plot_data[:, i], plot_data[:, j], bins=250, norm=norm)
            ax[i, j].set(xlabel=labels[i], ylabel=labels[j])
    fig.suptitle(title)
    fig.tight_layout()
    if filename != "":
        fig.savefig(filename, dpi=300)


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
    print(plot_data[0])

    # --- compute global histogram range ---
    bins = 250
    h_all = []

    for i in range(input_dim):
        for j in range(i):
            h, _, _ = np.histogram2d(plot_data[:, i], plot_data[:, j], bins=bins)
            h_all.append(h.ravel())

    h_all = np.concatenate(h_all)
    h_all = h_all[h_all > 0]  # avoid log(0)

    norm = LogNorm(vmin=h_all.min(), vmax=h_all.max())

    fig, ax = plt.subplots(ncols=input_dim, nrows=input_dim, figsize=(20, 20))
    for i in range(input_dim):
        for j in range(input_dim):
            if j > i:
                ax[i, j].set_axis_off()
                continue
            if j == i:
                try:
                    ax[i, j].hist(plot_data[:, i], bins=100)
                    ax[i, j].set(xlabel=labels[i], ylabel="Counts")
                except Exception:
                    try:
                        ax[i, j].hist(plot_data[:, i], bins=50)
                        ax[i, j].set(xlabel=labels[i], ylabel="Counts")
                    except Exception as e:
                        print(e)
                        ax[i, j].set_axis_off()
                continue
            ax[i, j].hist2d(plot_data[:, i], plot_data[:, j], bins=250, norm=norm)
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
    if filename != "":
        fig.savefig(filename, dpi=300)


def _shared_norm(x1, y1, x2, y2, bins):
    h1, _, _ = np.histogram2d(x1, y1, bins=bins)
    h2, _, _ = np.histogram2d(x2, y2, bins=bins)
    vmax = max(h1.max(), h2.max(), 1.0)
    return LogNorm(vmin=1, vmax=vmax)


def plot_two_datasets(data1, data2, title="", filename=""):
    labels = [
        "Weight [a.u]",
        "Energy [MeV]",
        "Time [ms]",
        "Direction x",
        "Direction y",
        "Direction z",
        "Position x [cm]",
        "Position y [cm]",
        "Position z [cm]",
        "Polarization x",
        "Polarization y",
        "Polarization z",
    ]

    if type(data1) is not np.ndarray:
        data1 = data1.numpy()
        data2 = data2.numpy()
        # Compute bounding box from data1
    lower = data1.min(axis=0)  # shape (7,)
    upper = data1.max(axis=0)  # shape (7,)
    # Boolean mask: points inside the 7D box
    mask = np.all((data2 >= lower) & (data2 <= upper), axis=1)
    # Filtered dataset
    data2 = data2[mask]

    input_dim = data1.shape[1]
    bins_2d = 250

    fig = plt.figure(figsize=(20, 20))

    # ----------- outer gridspec (room for annotations) -----------
    outer = GridSpec(1, 1, left=0.08, right=0.92, bottom=0.08, top=0.92, figure=fig)

    # ----------- inner gridspec for equal-sized plots -----------
    gs = outer[0].subgridspec(input_dim, input_dim, wspace=0.5, hspace=0.5)

    ax = np.empty((input_dim, input_dim), dtype=object)
    for i in range(input_dim):
        for j in range(input_dim):
            ax[i, j] = fig.add_subplot(gs[i, j])

    # ----------- plotting -----------
    for i in range(input_dim):
        for j in range(input_dim):
            if j > i:  # upper triangle (synthetic)
                norm = _shared_norm(
                    data1[:, i],
                    data1[:, j],
                    data2[:, j],
                    data2[:, i],
                    bins_2d,
                )
                ax[i, j].hist2d(data2[:, j], data2[:, i], bins=bins_2d, norm=norm)
                ax[i, j].set(xlabel=labels[j], ylabel=labels[i])

            elif j == i:  # diagonal
                try:
                    ax[i, j].hist(data1[:, i], bins=100, histtype="step", density=True)
                    ax[i, j].hist(data2[:, i], bins=100, histtype="step", density=True)
                    ax[i, j].set(xlabel=labels[i], ylabel="Density")
                except Exception:
                    try:
                        ax[i, j].clear()
                        ax[i, j].hist(
                            data1[:, i], bins=50, histtype="step", density=True
                        )
                        ax[i, j].hist(
                            data2[:, i], bins=50, histtype="step", density=True
                        )
                        ax[i, j].set(xlabel=labels[i], ylabel="Density")
                    except Exception as e:
                        print(e)
                        ax[i, j].set_axis_off()

            else:  # lower triangle (true)
                norm = _shared_norm(
                    data1[:, i],
                    data1[:, j],
                    data2[:, j],
                    data2[:, i],
                    bins_2d,
                )
                ax[i, j].hist2d(data1[:, i], data1[:, j], bins=bins_2d, norm=norm)
                ax[i, j].set(xlabel=labels[i], ylabel=labels[j])

    # ----------- annotations (guaranteed not clipped) -----------
    fig.text(
        0.80,
        0.95,
        "Upper triangle:\nSynthetic",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="orange", alpha=0.2),
    )

    fig.text(
        0.20,
        0.03,
        "Lower triangle:\nTrue data",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="blue", alpha=0.2),
    )

    # ----------- upper‑triangle zig‑zag outline (figure coords) -----------
    N = input_dim
    eps = 0.048
    eps_y = 0.04
    extra_off = 0.005

    off_x = 0.045
    off_y = 0.93
    xs, ys = [], []

    xs += [0.93, off_x + 1 / N * 0.84]
    ys += [0.93, 0.93]

    for i in range(N - 1):
        x_left = off_x + (i + 1 + i * eps) / N * 0.84
        x_right = off_x + (i + 2 + (i + 1) * eps) / N * 0.84

        y = off_y - extra_off - (i + 1 + i * eps_y) / N * 0.84

        xs += [x_left, x_right]
        ys += [y, y]
    xs += [x_right, 0.93]
    ys += [y, y]
    xs += [0.93, 0.93]
    ys += [y, off_y]

    fig.add_artist(
        Line2D(
            xs,
            ys,
            transform=fig.transFigure,
            lw=2,
            color="orange",
            zorder=10,
        )
    )

    N = input_dim
    eps = 0.0465
    eps_y = 0.04
    extra_off = 0.005

    off_x = -0.08
    off_y = 0.93
    min_y = 0.05
    xs, ys = [], []

    # xs += [0.93, off_x + 1/N * 0.84]
    # ys += [0.93, 0.93]

    for i in range(N - 1):
        x_left = off_x + (i + 1 + i * eps) / N * 0.84
        x_right = off_x + (i + 2 + (i + 1) * eps) / N * 0.84

        y = off_y - extra_off - (i + 1 + i * eps_y) / N * 0.84

        xs += [x_left, x_right]
        ys += [y, y]
    xs += [x_right, x_right]
    ys += [y, min_y]
    xs += [x_right, 0.03999]
    ys += [min_y, min_y]
    xs += [0.03999, 0.03999]
    ys += [min_y, off_y - extra_off - 1 / N * 0.84]

    fig.add_artist(
        Line2D(
            xs,
            ys,
            transform=fig.transFigure,
            lw=2,
            color="blue",
            zorder=10,
        )
    )

    sm = ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cax = fig.add_axes([0.945, 0.15, 0.015, 0.7])

    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Intensity [a.u]")

    # ----------- title -----------
    fig.suptitle(title, fontsize=16, y=0.99)

    if filename:
        fig.savefig(filename, dpi=300)

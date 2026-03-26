import matplotlib.pyplot as plt


def plot_correlations_7d(data, title=""):
    labels = [
        "Weight",
        "Energy",
        "Direction x",
        "Direction y",
        "Direction z",
        "Position x",
        "Position y",
    ]
    # labels = ["Angle theta", "Angle phi", "Position x", "Position y"]
    input_dim = 7
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

            ax[i, j].hist2d(data[:, i], data[:, j], bins=100)
            ax[i, j].set(xlabel=labels[i], ylabel=labels[j])
    fig.suptitle(title)
    fig.tight_layout()


def plot_correlations_6d(data, title=""):
    labels = [
        "Weight",
        "Energy",
        "Theta",
        "Phi",
        "Position x",
        "Position y",
    ]
    input_dim = 6
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

            ax[i, j].hist2d(data[:, i], data[:, j], bins=100)
            ax[i, j].set(xlabel=labels[i], ylabel=labels[j])

    fig.suptitle(title)
    fig.tight_layout()


import numpy as np
import joblib
import matplotlib.pyplot as plt
# ==============================================================================
# ================ Make N synthetic samples, and plot their correlations
# ==============================================================================


def plot_correlations(filename="kde_model.pkl"):
    labels = [
        "Weight",
        "Energy",
        "Direction x",
        "Direction y",
        "Direction z",
        "Position x",
        "Position y",
    ]
    kde = joblib.load(filename)
    samples = kde.sample(100000)
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
                    ax[i, j].hist(samples[:, i], bins=50)
                    ax[i, j].set(xlabel=labels[i], ylabel="Counts")
                except Exception as e:
                    print(e)
                    ax[i, j].set_axis_off()
                continue
            ax[i, j].hist2d(samples[:, i], samples[:, j], bins=100)
            ax[i, j].set(xlabel=labels[i], ylabel=labels[j])
    fig.tight_layout()



plot_correlations()

plt.show()


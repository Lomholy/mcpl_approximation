import numpy as np
from kde_definition import AdaptiveKDE
import joblib
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from plotting import  plot_correlations_7d
from data_load import postprocess, save_data_as_mcpl
# ==============================================================================
# ================ Make N synthetic samples, and plot their correlations
# ==============================================================================


xmin, ymin, dx, dy = np.load("../meta_params_dist.npy")
lims = np.load("../limits.npy")
mins = torch.asarray(lims[0])
maxs = torch.asarray(lims[1])
kde = joblib.load("kde.pkl")
n_samples = 1_000_000
samples = []
while len(samples) < n_samples:
    # Limit samples to within the preprocessed data limits

    batch = torch.asarray(kde.sample(10_000))

    batch = postprocess(batch, xmin, ymin, dx, dy)
    mask = (batch >= mins) & (batch <= maxs)
    batch = batch[mask.all(axis=1)]
    samples = samples + batch.tolist()
samples = samples[:n_samples]
samples = torch.asarray(samples)

print(f"Number of samples plotted {samples.shape[0]}")
plot_correlations_7d(samples, "AKDE")

save_data_as_mcpl(samples, "../kde_samples")


plt.show()


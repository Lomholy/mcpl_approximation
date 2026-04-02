import numpy as np
from kde_definition import AdaptiveKDE
import joblib
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from plotting import plot_correlations_5d
from data_load import postprocess
# ==============================================================================
# ================ Make N synthetic samples, and plot their correlations
# ==============================================================================


xmin, ymin, dx, dy = np.load("../meta_params_dist.npy")
lims = np.load("../prep_data_limits.npy")

kde = joblib.load("kde.pkl")
samples = torch.asarray(kde.sample(100000))
# Limit samples to within the preprocessed data limits

mins = torch.asarray(lims[0])
maxs = torch.asarray(lims[1])

mask = (samples >= mins) & (samples <= maxs)

samples = samples[mask.all(axis=1)]

plot_correlations_5d(samples, "Synthetic: Preprocessed data")
samples = postprocess(samples, xmin, ymin, dx, dy)
plot_correlations_5d(samples, "Synthetic: Postprocessed data")

plt.show()


import numpy as np
from adaptive_kde import AdaptiveKDE
import joblib
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from plotting import plot_correlations_6d
from data_load import postprocess
# ==============================================================================
# ================ Make N synthetic samples, and plot their correlations
# ==============================================================================


xmin, ymin, dx, dy = np.load("../meta_params_dist.npy")

kde = joblib.load("./adaptive_kde.pkl")
samples = torch.asarray(kde.sample(100000))

post_data = postprocess(samples, xmin, ymin, dx, dy)

plot_correlations_6d(post_data, "Synthetic: Postprocessed data")

plt.show()


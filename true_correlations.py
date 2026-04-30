import matplotlib.pyplot as plt
import argparse
import numpy as np
from data_load import (
    load_mcpl_file,
    preprocess,
    preprocess_nn,
    postprocess,
    postprocess_nn,
    dim_reduction,
)
from plotting import plot_correlations_5d, plot_correlations_6d, plot_correlations_7d


parser = argparse.ArgumentParser()


parser.add_argument("--n_samples", default=1e5)
parser.add_argument("--raw", action="store_true")
parser.add_argument("--dim_red", action="store_true")
parser.add_argument("--preprocess", action="store_true")
parser.add_argument("--preprocess_nn", action="store_true")
parser.add_argument("--postprocess", action="store_true")
parser.add_argument("--postprocess_nn", action="store_true")

arg = parser.parse_args()


data = load_mcpl_file("ODIN.mcpl.gz", int(arg.n_samples))
# data = load_mcpl_file("kde_samples.mcpl.gz", int(arg.n_samples))
if arg.raw:
    plot_correlations_7d(data, "TRUE: Raw correlations")
final_limits = np.zeros((2,7))
for i in range(7):
    final_limits[0,i] = data[:,i].min()
    final_limits[1,i] = data[:,i].max()
np.save("limits.npy", final_limits)

dim_red = dim_reduction(data)
if arg.dim_red:
    plot_correlations_6d(dim_red, "TRUE: Dimensionally reduced data", filename="dim_red.png")
    plot_correlations_6d(dim_red, filename="dim_red.png")


prep, xmin, ymin, dx, dy = preprocess(data)
np.save("meta_params_dist.npy", np.array([xmin, ymin, dx, dy]))
# Get preprocessed limits and save them in a file as well
lims = np.zeros((2, prep.shape[1]))
for i in range(prep.shape[1]):
    lims[0, i] = prep[:, i].min()
    lims[1, i] = prep[:, i].max()
np.save("prep_data_limits.npy", np.array(lims))


if arg.preprocess:
    plot_correlations_6d(prep, "TRUE: Preprocessed data")

prep_nn, mins, dxs = preprocess_nn(data)
lims = np.zeros((2, prep.shape[1]))
for i in range(prep.shape[1]):
    lims[0, i] = prep_nn[:, i].min()
    lims[1, i] = prep_nn[:, i].max()
np.save("prep_data_limits_nn.npy", np.array(lims))
if arg.preprocess_nn:
    plot_correlations_6d(prep_nn, "TRUE: Preprocessed_nn data")

post = postprocess(prep, xmin, ymin, dx, dy)
if arg.postprocess:
    plot_correlations_7d(post, "TRUE: Postprocessed data")

post_nn = postprocess_nn(prep_nn, mins, dxs)
if arg.postprocess_nn:
    plot_correlations_7d(post_nn, "TRUE: Postprocessed_nn data")

plt.show()

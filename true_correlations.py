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
from plotting import plot_correlations_6d, plot_correlations_7d


arg = argparse.ArgumentParser()


arg.add_argument("--raw", action="store_true")
arg.add_argument("--dim_red", action="store_true")
arg.add_argument("--preprocess", action="store_true")
arg.add_argument("--preprocess_nn", action="store_true")
arg.add_argument("--postprocess", action="store_true")
arg.add_argument("--postprocess_nn", action="store_true")

parser = arg.parse_args()


data = load_mcpl_file("ODIN.mcpl.gz", 10)
if parser.raw:
    plot_correlations_7d(data, "TRUE: Raw correlations")

dim_red = dim_reduction(data)
if parser.dim_red:
    plot_correlations_6d(dim_red, "TRUE: Dimensionally reduced data")


prep, xmin, ymin, dx, dy = preprocess(data)
np.save("meta_params_dist.npy", np.array([xmin, ymin, dx, dy]))

if parser.preprocess:
    plot_correlations_6d(prep, "TRUE: Preprocessed data")

prep_nn, mins, dxs = preprocess_nn(data)
if parser.preprocess_nn:
    plot_correlations_6d(prep_nn, "TRUE: Preprocessed_nn data")

post = postprocess(prep, xmin, ymin, dx, dy)

if parser.postprocess:
    plot_correlations_6d(post, "TRUE: Postprocessed data")

post_nn = postprocess_nn(prep_nn, mins, dxs)
if parser.postprocess_nn:
    plot_correlations_6d(post_nn, "TRUE: Postprocessed_nn data")


plt.show()

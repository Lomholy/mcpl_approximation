import numpy as np
import sys
sys.path.append("..")
from plotting import plot_correlations_7d
import matplotlib.pyplot as plt




data = np.loadtxt("./C_samples.csv", delimiter=",")


plot_correlations_7d(data, "C Generated samples")


plt.show()

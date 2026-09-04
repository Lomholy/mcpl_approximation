import numpy as np
import sys
sys.path.append("..")
from plotting import plot_correlations_7d, plot_two_datasets
import matplotlib.pyplot as plt
from data_load import load_mcpl_file
import torch




C_gen = torch.asarray(np.loadtxt("./C_samples.csv", delimiter=","))


input = torch.load("../gaussian_input.pkl")
C_gen = C_gen[:input.shape[0]]

print(C_gen.shape, input.shape)
plot_two_datasets(input, C_gen, title="Input vs Samples from C program", filename="C_prog_Gauss.png")


plt.show()

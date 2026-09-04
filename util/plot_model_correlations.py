# Author: Daniel Lomholt Christensen @UCPH 2026
# A script that plots the results of the three different models, and the true result
from data_load import load_mcpl_file
from plotting import plot_correlations_7d, plot_two_datasets
import matplotlib.pyplot as plt
import torch



cfm = "./mvp_cmf_samples.mcpl.gz"
vae = "./vae_samples.mcpl.gz"
kde = "./kde_samples.mcpl.gz"
odin = "./ODIN.mcpl.gz"
c_gen = "./C_gen.mcpl.gz"

n_samples = 1_000_000
cfm = load_mcpl_file(cfm, n_samples)
c_gen = load_mcpl_file(c_gen, n_samples)

# gauss_cfm = torch.load("./mvp_gaussian_output.pkl")
gauss_vae = torch.load("VAE/gaussian_output.pkl")
# kde = load_mcpl_file(kde, n_samples)
# odin = load_mcpl_file(odin, n_samples)
gauss_odin = torch.load("./gaussian_input.pkl")
print(gauss_odin.shape, gauss_vae.shape)


# plot_correlations_7d(cfm, "CFM correlations", filename="cfm_corr.png")
# plot_correlations_7d(vae, "VAE correlations", filename="vae_corr.png")
# plot_correlations_7d(kde, "KDE correlations", filename="kde_corr.png")
# plot_correlations_7d(odin, "Simulation correlations", filename="sim_corr.png")

# plot_two_datasets(odin, cfm, title="Input vs CFM", filename="MVP ODIN_CFM_comparison.png")
# plot_two_datasets(gauss_odin, gauss_cfm, title="Input vs CFM in Gaussian space", filename="MVP ODIN_CFM_gauss_comparison.png")
# plot_two_datasets(gauss_odin, gauss_cfm, title="Input vs CFM in Gaussian space",)
plot_two_datasets(gauss_odin, gauss_vae, title="Input vs VAE in Gaussian space",)



# plot_two_datasets(odin, c_gen, title="Input vs C generated mcpl", filename="input_v_C.png")

plt.show()


import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append("..")
from data_load import load_mcpl_file, save_data_as_mcpl
import numpy as np
import copy

def split_mcpl_file(mcpl_file, requested_sizes, filename, input_particles, original_data):
    data = load_mcpl_file(mcpl_file, n_particles=input_particles).numpy()

    print(data.shape)
    total_I = original_data[:,0].sum()
    print(f"Original data Intensity = {total_I}")

    for size in requested_sizes:
        size = int(size)
        idx = np.random.randint(0,data.shape[0], size)
        new_data = copy.copy(data)
        diff = total_I - data[idx,0].sum()

        new_data[idx, 0] *= diff/len(idx)
        print(f"size={size}\ntotal I of new data = {new_data[idx,0].sum()}")
        save_data_as_mcpl(new_data[idx], filename=filename + f"_{size}")



if __name__ == "__main__":
    original_data = load_mcpl_file("../ODIN.mcpl.gz", n_particles=254_134_779).numpy()
    requested_sizes = np.logspace(1, 6, 6)
    split_mcpl_file("../ODIN.mcpl.gz", requested_sizes, "./mcpl_files/input", 1_000_000, original_data)
    print("\n")
    split_mcpl_file("../cmf_samples.mcpl.gz", requested_sizes, "./mcpl_files/cfm", 10_000_000, original_data)





import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append("..")
from data_load import load_mcpl_file, save_data_as_mcpl
import numpy as np
import copy


def split_mcpl_file(mcpl_file, requested_sizes, filename, original_data=None):
    data = load_mcpl_file(mcpl_file).numpy()

    if original_data is None:
        original_data = data

    print(data.shape)
    total_N = original_data.shape[0]
    total_I = original_data[:,0].sum()
    print(f"Original data Intensity = {total_I}")

    for size in requested_sizes:
        if size>data.shape[0]:
            continue
        size = int(size)
        idx = np.random.choice(data.shape[0], size, replace=False)
        new_data = copy.copy(data[idx])
        scale = total_N/size
        new_data[:, 0] *= scale
        print(f"size={size}\ntotal I of new data = {new_data[:,0].sum()}")
        save_data_as_mcpl(new_data, filename=filename + f"_{size}")
    return


if __name__ == "__main__":
    requested_sizes = np.geomspace(100, 25_416_962, 70, dtype=int)
    original_data = load_mcpl_file("../ODIN.mcpl.gz").numpy()
    split_mcpl_file("../ODIN.mcpl.gz", requested_sizes, "./mcpl_files/input")
    split_mcpl_file("../ODIN_n11.mcpl.gz", requested_sizes, "./mcpl_files/big")





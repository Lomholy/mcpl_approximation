# Author: Daniel Lomholt Christensen
import joblib
import mcpl
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split


def load_mcpl_file(filepath, n_blocks):
    mcplfile = mcpl.MCPLFile(filepath)
    data = np.zeros([n_blocks * 10000, 7], dtype=np.float32)

    for i, p in enumerate(mcplfile.particle_blocks):
        data[i * 10000 : (i + 1) * 10000, 0] = np.asarray(p.weight)
        data[i * 10000 : (i + 1) * 10000, 1] = np.asarray(p.ekin)
        data[i * 10000 : (i + 1) * 10000, 2:5] = np.asarray(
            np.array([p.ux, p.uy, p.uz])
        ).T
        data[i * 10000 : (i + 1) * 10000, 5:] = np.asarray(
            np.array([p.x, p.y])
        ).T

        if i == n_blocks - 1:
            break

    return data


# Load data
data = load_mcpl_file("../ODIN.mcpl.gz", 10)

# Split into training / validation
train, val = train_test_split(data, test_size=0.1, shuffle=True, random_state=0)

# Parameter grid
kernels = ["gaussian", "tophat", "epanechnikov"]
bandwidths = np.logspace(-4, 0, 10)

best_score = -np.inf
best_params = None
best_kde = None

# Grid search
for kernel in kernels:
    for bw in bandwidths:
        kde = KernelDensity(kernel=kernel, bandwidth=bw)
        kde.fit(train)

        score = kde.score(val)  # average log-likelihood

        if score > best_score:
            best_score = score
            best_params = (kernel, bw)
            best_kde = kde
            print(f"New best: kernel={kernel}, bw={bw}, score={score}")


# Save best model
print("Best parameters:", best_params)
joblib.dump(best_kde, "kde_model.pkl")
print("Saved model as kde_model.pkl")

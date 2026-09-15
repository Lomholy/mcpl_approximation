import mcstasscript as ms
import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# LOAD IN DATA
# ==============================================================================
det = -1
original_data = ms.load_data(f"simulations/raw_2")[det]
og_int = original_data.Intensity


results = {"input": [[], []], "big": [[], []], "fm": [[], []], "raw": [[], []]}

for run in os.listdir("simulations"):
    if os.path.isfile(run):# or run.startswith("raw_2"):
        continue
    print(run)
    data = ms.load_data(f"simulations/{run}")[det]
    intensity = data.Intensity
    error = data.Error
    loss = np.sum((og_int - intensity) ** 2)
    name = run.split("_")[0]
    results[name][0].append(int(run.split("_")[1]))
    results[name][1].append(loss)

# ==============================================================================
# PLOT RESULTS
# ==============================================================================
fig, ax = plt.subplots()
ax.plot(
    results["input"][0],
    results["input"][1],
    "+b", alpha=0.8,
    label="McStas MCPL 10^6 L2",
)

ax.plot(
    results["fm"][0],
    results["fm"][1],
    "+r", alpha=0.8,
    label="CFM MCPL 10^7 L2",
)
ax.plot(
    results["big"][0],
    results["big"][1],
    "+g", alpha=0.8,
    label="McStas MCPL 2.5*10^7 L2",
)

# for i in range(4):
#     print(f"Raw number {results["raw"][0][i]}")
#     print(f"Loss is {results["raw"][1][i]:.2g}")
ax.hlines(y=results["raw"][1], xmin=0, xmax=25_700_000, linestyles=["--"], label="Target loss")
ax.vlines(x=1_000_000, ymin=1e8, ymax=1e15, color="purple",linestyles=["--"], label="Trained particle count")
ax.legend()
ax.set(
    yscale="log", ylim=(1e8,1e15), xscale="log", xlabel="Number of particles [#]", ylabel="MSE [dI**2]"
)
ax.grid(True)

fig.tight_layout()

fig.savefig("MSE.png")

plt.show()

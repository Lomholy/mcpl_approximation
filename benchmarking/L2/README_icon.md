# PSI_ICON L2 benchmark

Compares the PSI_ICON MCPL file, and a CFM model trained on it, against an
independent reference MCPL file. The reference is a second run of the same
instrument with another random seed.

## Input files

Both MCPL files record the neutrons just before the sample position of
`PSI_ICON.instr` (McStas `mcstas-comps/examples/PSI/PSI_ICON`). Insert an
`MCPL_output` component before `SampleIn` and run with 1e9 neutrons:

```
COMPONENT mcpl_before_sample = MCPL_output(filename="PSI_ICON_before_sample")
AT (0,0,-Zdepth/2.0-1e-9) RELATIVE Goniometer
```

Run it twice with different `--seed` values. The first file is the original
(`data_files/mcpl_files/PSI_ICON.mcpl.gz`) that the model is trained on, the
second is the reference (`data_files/mcpl_files/PSI_ICON_ref.mcpl.gz`). Both
must be written with the same `weight_mode`.

## Model

Train and export with the existing CFM scripts. `--random_subset` draws the
training neutrons at random from the file, and the weights are rescaled so the
subset carries the summed weight of the whole file:

```
cd models/CFM
python train.py --input_mcpl ../../data_files/mcpl_files/PSI_ICON.mcpl.gz --random_subset
python eval.py
```

`Source_ML` runs the model on CUDA or MPS when available (`device="auto"`),
which is much faster than the CPU. `icon_sample.instr` exposes it as
`ml_device`.

## Benchmark

`icon_sample.instr` places the MCPL source (or `Source_ML`) before a Fresnel
zone plate (`FZP_simple`) and records the signal after it, on a PSD flush
against the plate and on a PSD in the focal plane of the design wavelength.

```
cd benchmarking/L2
python icon_l2_benchmark.py --ref-mcpl ../../data_files/mcpl_files/PSI_ICON_ref.mcpl.gz --out-dir <output dir>
```

The benchmark runs the reference file through `icon_sample.instr`, then, for a
range of particle counts n, runs a random subset of n neutrons of the original
file (weights rescaled by N / n) and the `Source_ML` component with ncount = n.
The loss is the sum of squared pixel differences to the reference PSD. The
PSDs of the full runs and their differences to the reference are written to
the output directory too (`icon_psd_compare.py`).

On macOS `mcrun` may need `MCSTAS_CC_OVERRIDE=/usr/bin/clang`, see
`mcstas_comps/README.md`.

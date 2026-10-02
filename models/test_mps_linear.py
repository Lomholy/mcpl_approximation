"""Check that the CFM and VAE linear layers give the same result on the CPU and on MPS.

Run from models/:  python test_mps_linear.py
"""
import importlib.util
import sys
from pathlib import Path

import torch


def load_model_module(name):
    path = Path(__file__).resolve().parent / name / "model.py"
    spec = importlib.util.spec_from_file_location(f"{name}_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if not torch.backends.mps.is_available():
    print("MPS not available, nothing to check")
    sys.exit(0)

torch.manual_seed(0)
x = torch.randn(1000, 12)
t = torch.rand(1000, 1)

cfm = load_model_module("CFM").VelocityField().eval()
for p in cfm.out[-1].parameters():
    torch.nn.init.normal_(p, std=0.1)
cfm_mps = load_model_module("CFM").VelocityField().eval()
cfm_mps.load_state_dict(cfm.state_dict())
cfm_mps.to("mps")
with torch.no_grad():
    diff = (cfm(x, t) - cfm_mps(x.to("mps"), t.to("mps")).cpu()).abs().max().item()
print(f"CFM VelocityField cpu vs mps max difference: {diff:.3g}")
assert diff < 1e-3

lin = load_model_module("CFM").Linear(12, 64)
lin_mps = load_model_module("CFM").Linear(12, 64)
lin_mps.load_state_dict(lin.state_dict())
lin_mps.to("mps")
with torch.no_grad():
    diff = (lin(x) - lin_mps(x.to("mps")).cpu()).abs().max().item()
print(f"CFM Linear cpu vs mps max difference: {diff:.3g}")
assert diff < 1e-4

"""Exports a smoke-test ONNX model for Test_Source_ML.instr's backend="onnx"
path. Replaces the dead mcstas_comps/onnx_implementation/onnx_test.py (which
referenced a now-nonexistent ../NF/nf_definition module from an earlier
prototype) with one that mirrors export_torch_smoke.py's structure/model."""
import os
import sys

import torch

sys.path.append("../models/CFM")
sys.path.append("../utils")

from model import Sampler, VelocityField  # noqa: E402
from data_load import export_model_as_onnx  # noqa: E402
from make_smoke_transformer import write_smoke_transformer  # noqa: E402

if __name__ == "__main__":
    device = "cpu"

    checkpoint_path = "../data_files/models/CFM.pth"
    velocity = VelocityField()

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        velocity.load_state_dict(ckpt["state_dict"])
    else:
        print(
            f"⚠️  No trained checkpoint found at {checkpoint_path}; "
            "exporting an untrained model for pipeline smoke-testing only."
        )

    transformer_path = "../data_files/preprocess/gaussian_transformer.bin"
    if not os.path.exists(transformer_path):
        print(
            f"⚠️  No trained transformer found at {transformer_path}; "
            "generating a synthetic one for pipeline smoke-testing only."
        )
        transformer_path = write_smoke_transformer("gaussian_transformer_smoke.bin")

    velocity = velocity.to(device)
    sampler = Sampler(velocity, n_steps=64, device=device)

    onnx_path = "CFM_sampler.onnx"
    export_model_as_onnx(
        sampler,
        onnx_path,
        device=device,
        transformer_file_path=transformer_path,
    )

    print(f"wrote {onnx_path}")

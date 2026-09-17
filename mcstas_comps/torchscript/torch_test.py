import os
import sys

import torch

sys.path.append("../../models/CFM")
sys.path.append("../../utils")

from model import Sampler, VelocityField  # noqa: E402
from data_load import export_model_as_torchscript  # noqa: E402

if __name__ == "__main__":
    device = "cpu"

    checkpoint_path = "../../data_files/models/CFM.pth"
    velocity = VelocityField()

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        velocity.load_state_dict(ckpt["state_dict"])
        n_training_samples = 1_000_000
    else:
        print(
            f"⚠️  No trained checkpoint found at {checkpoint_path}; "
            "exporting an untrained model for pipeline smoke-testing only."
        )
        n_training_samples = 0

    velocity = velocity.to(device)
    sampler = Sampler(velocity, n_steps=64, device=device)

    torch_path = "CFM_sampler.pt"
    export_model_as_torchscript(
        sampler, torch_path, device=device, n_training_samples=n_training_samples
    )

    # Round-trip check, mirroring what Source_ML_torch.comp's C wrapper will do.
    extra_files = {"n_training_samples": ""}
    reloaded = torch.jit.load(torch_path, _extra_files=extra_files)
    print("n_training_samples metadata:", extra_files["n_training_samples"])

    x = torch.randn(4, 12, device=device)
    out = reloaded(x)
    print("sample output shape:", tuple(out.shape))

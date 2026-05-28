import sys
sys.path.append("../NF/")

import torch
import onnx
from nf_definition import VelocityField


if __name__ == "__main__":
    device = "cpu"
    ckpt = torch.load("../NF/flowModel.pth", map_location=device)
    nf = VelocityField().to(device)
    nf.load_state_dict(ckpt["state_dict"])
    nf.eval()
    dummy_input = torch.randn(1, 7).to(device) 
    dummy_t = torch.randn(1, 1).to(device) 
    # Export ONNX with external weights
    onnx_path = "velocity_field.onnx"

    prog = torch.onnx.export(
        nf,
        (dummy_input, dummy_t),
        external_data=True,
        export_params=False,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
        dynamic_shapes={
            "x": {0: "batch"},
            "t": {0: "batch"},
        }
    )
    prog.save(onnx_path, external_data=True)
    # Convert to external data format (separate weights file)
    model = onnx.load(onnx_path)

    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="velocity_field_weights.bin",
        size_threshold=1024,  # bytes threshold before moving tensors outside
    )

    print("✅ Export complete!")


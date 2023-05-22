from models import PointNetReg
import torch
import torch.nn as nn

model = PointNetReg(num_classes=6)
model.load_state_dict(torch.load("PointNetReg.pth", "cuda"))
model.eval()

input_size_tuple = ((10, 15, 10000),)
x = tuple(
        [torch.randn(i) for i in input_size_tuple]
    )

torch.onnx.export(
    model,
    args=x,  # model input (or a tuple for multiple inputs)
    f="PointNetReg.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=16,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch", 2: "points_num"},
        "output": {0: "batch", 1: "points_num"},
    }
)

import torch
from models import MeshSegNet

model = MeshSegNet()
model.load_state_dict(torch.load("MeshSegNet_Max_15_classes_72samples_lr1e-2_best.zip", map_location="cuda")["model_state_dict"])
model.eval()

input_size_tuple = ((10, 15, 10000),(10, 10000, 10000),(10, 10000, 10000))
x = tuple(
        [torch.randn(i) for i in input_size_tuple]
    )

torch.onnx.export(
    model,
    args=x,  # model input (or a tuple for multiple inputs)
    f="MeshSegNet.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=16,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input", "a_s", "a_l"],  # the model's input names
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch", 2: "points_num"},
        "a_s": {0: "batch", 1: "points_num", 2: "points_num"},
        "a_l": {0: "batch", 1: "points_num", 2: "points_num"},
        "output": {0: "batch", 1: "points_num"},
    },
)
from utils.getmodel import get_model
import torch
import netron

model=get_model("unresnet_noInit_inception")
x = torch.rand((1, 1, 40, 40))
o = model(x)
onnx_path = "test.onnx"
torch.onnx.export(model, x, onnx_path, opset_version=11)
netron.start(onnx_path)

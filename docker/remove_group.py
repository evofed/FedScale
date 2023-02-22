import pickle
import torch.nn as nn
import torch
import sys
from argparse import Namespace
from fedscale.core.net2netlib import set_model_layer, get_model_layer
from fedscale.core.model_manager import SuperModel
from thop import profile

model_path = sys.argv[1]


with open(model_path, 'rb') as f:
    model = pickle.load(f)

ns = Namespace(**{"task": "vision", "data_set": "cifar10"})

super_model = SuperModel(model, ns, 0)
layers = super_model.get_weighted_layers()
print(layers)

macs, params = profile(model, inputs=(torch.randn(10, 3, 32, 32),), verbose=False)
print(f"prev: {macs}, {params}")

for _, layer_name in layers:
    layer = get_model_layer(model, layer_name)
    if isinstance(layer, torch.nn.Conv2d):
        if layer.groups != 1:
            new_layer = nn.Conv2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=1,
                bias=True if layer.bias is not None else False
            )
            set_model_layer(model, new_layer, layer_name)

macs, params = profile(model, inputs=(torch.randn(10, 3, 32, 32),), verbose=False)
print(f"after: {macs}, {params}")

with open(model_path+'.new', 'wb') as f:
    pickle.dump(model, f)
import torch
from fedscale.core.net2netlib import get_model_layer
from typing import Iterable

def get_neuron_weight_diff(client_model: torch.nn.Module, server_model: torch.nn.Module, layers: Iterable[str]):
    layer_diffs = {}
    for layer in layers:
        client_layer = get_model_layer(client_model, layer)
        server_layer = get_model_layer(server_model, layer)
        if isinstance(client_layer, torch.nn.Conv2d) and isinstance(server_layer, torch.nn.Conv2d):
            client_weight = client_layer.weight
            server_weight = server_layer.weight
            diff = torch.amax(torch.abs(client_weight - server_weight)/torch.abs(server_weight), dim=(1,2,3))
        elif isinstance(client_layer, torch.nn.Linear) and isinstance(server_layer, torch.nn.Linear):
            client_weight = client_layer.weight
            server_weight = server_layer.weight
            diff = torch.amax(torch.abs(client_weight - server_weight)/torch.abs(server_weight), dim=1)
        elif isinstance(client_layer, torch.nn.BatchNorm2d) and isinstance(server_layer, torch.nn.BatchNorm2d):
            client_weight = client_layer.weight
            server_weight = server_layer.weight
            diff = torch.abs(client_weight - server_weight)/torch.abs(server_weight)
        else:
            raise NotImplementedError(f"Layer {layer} is not supported")
        layer_diffs[layer] = diff
    return layer_diffs
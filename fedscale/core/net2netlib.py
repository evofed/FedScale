from copy import deepcopy
from typing import OrderedDict
import numpy as np
import torch
import collections

import logging

def get_model_layer_weight(torch_model, attri_name: str):
    layer = get_model_layer(torch_model, attri_name)
    return layer.weight.data


def get_model_layer_grad(torch_model, attri_name: str):
    layer = get_model_layer(torch_model, attri_name)
    return layer.weight.grad.data


def get_model_layer(torch_model, attri_name: str):
    if '.' in attri_name:
        attri_names = attri_name.split('.')
    else:
        attri_names = [attri_name]
    module = torch_model
    for attri in attri_names:
        module = getattr(module, attri)
    return module


def set_model_layer(torch_model, torch_module, attri_name: str):
    if '.' in attri_name:
        attri_names = attri_name.split('.')
    else:
        attri_names = [attri_name]
    to_modify = attri_names[-1]
    attri_name = ".".join(attri_names[:-1])
    if len(attri_name) != 0:
        module = get_model_layer(torch_model, attri_name)
        setattr(module, to_modify, torch_module)
    else:
        setattr(torch_model, to_modify, torch_module)


def widen_child_fc_helper(params: OrderedDict, mapping, noise_factor=5e-2):
    """
    weights have shape: (out_channels, in_channels)
    bias has shape: (out_channels, )
    """
    new_params = collections.OrderedDict()
    weights = params['weight'].numpy()
    out_channel, _ = weights.shape
    new_in_channel = len(mapping)
    new_weights = np.zeros((out_channel, new_in_channel))
    scale = [mapping.count(m) for m in mapping]
    for i in range(len(mapping)):
        new_weights[:, i] = weights[:, mapping[i]].copy() / scale[i]
    new_params['weight'] = torch.from_numpy(new_weights)
    new_params['bias'] = deepcopy(params['bias'])
    return new_params

def widen_parnet_conv_helper(params: OrderedDict, mapping, noise_factor=5e-2):
    """
    weights have shape: (out_channels, in_channels, kernel_height, kernel_width)
    bias has shape: (out_channels, )
    """
    new_params = collections.OrderedDict()
    weights = params['weight'].numpy()
    _, in_channel, kernel_height, kernel_width = weights.shape
    new_out_channel = len(mapping)
    new_weights = np.zeros((new_out_channel, in_channel, kernel_height, kernel_width))
    for i in range(len(mapping)):
        new_weights[i, :, :, :] = weights[mapping[i], :, :, :].copy() \
            + np.random.normal(scale=noise_factor*weights[mapping[i], :, :, :].std(),
            size=list(weights[mapping[i], :, :, :].shape))
    new_params['weight'] = torch.from_numpy(new_weights)
    if 'bias' in params.keys():
        bias = params['bias'].numpy()
        new_bias = np.zeros((new_out_channel))
        for i in range(len(mapping)):
            new_bias[i] = bias[mapping[i]].copy() \
                + np.random.normal(scale=noise_factor, size=list(bias[mapping[i]].shape))
        new_params['bias'] = torch.from_numpy(new_bias)
    return new_params

def widen_child_conv_helper(params: OrderedDict, mapping, noise_factor=5e-2):
    """
    weights have shape: (out_channels, in_channels, kernel_height, kernel_width)
    bise have shape: (out_channels, )
    """
    new_params = collections.OrderedDict()
    weights = params['weight'].numpy()
    out_channel, _, kernel_height, kernel_width = weights.shape
    new_in_channel = len(mapping)
    new_weights = np.zeros((out_channel, new_in_channel, kernel_height, kernel_width))
    scale = [mapping.count(m) for m in mapping]
    for i in range(len(mapping)):
        new_weights[:, i, :, :] = weights[:, mapping[i], :, :] / scale[i]
    new_params['weight'] = torch.from_numpy(new_weights)
    if 'bias' in params.keys():
        new_params['bias'] = deepcopy(params['bias'])
    return new_params

def shrink_conv_helper(params: OrderedDict, new_in_channel, new_out_channel ,noise_Factor=5e-2):
    """
    weights have shape: (out_channels, in_channels, kernel_height, kernel_width)
    bise have shape: (out_channels, )
    """
    new_params = collections.OrderedDict()
    weights = params['weight'].numpy()
    out_channel, in_channel, kernel_height, kernel_width = weights.shape
    new_weights = np.zeros((new_out_channel, new_in_channel, kernel_height, kernel_width))
    out_bound = min(new_out_channel, weights.shape[0])
    in_bound = min(new_in_channel, weights.shape[1])

    new_weights[:out_bound, :in_bound, :, :] = deepcopy(weights[:out_bound, :in_bound, :, :])
    new_params['weight'] = torch.from_numpy(new_weights)
    if 'bias' in params.keys():
        new_params['bias'] = torch.zeros(new_out_channel)
        new_params['bias'][:out_bound] = deepcopy(params['bias'][:out_bound])
    return new_params

def widen_batch_helper(batch: OrderedDict, mapping, noise_factor=5e-2):
    new_batch = collections.OrderedDict()
    for param_name in batch.keys():
        batch[param_name] = batch[param_name].numpy()
        if param_name in ['num_batches_tracked']:
            new_batch[param_name] = deepcopy(torch.from_numpy(batch[param_name]))
        else:
            new_batch[param_name] = np.zeros((len(mapping), ))
            for i in range(len(mapping)):
                new_batch[param_name][i] = \
                    batch[param_name][mapping[i]].copy() + np.random.normal(scale=noise_factor, size=list(batch[param_name][mapping[i]].shape))
            new_batch[param_name] = torch.from_numpy(new_batch[param_name])
    return new_batch

def shrink_batch_helper(batch: OrderedDict, new_num_features, noise_factor=5e-2):
    new_batch = collections.OrderedDict()
    for param_name in batch.keys():
        batch[param_name] = batch[param_name].numpy()
        if param_name in ['num_batches_tracked']:
            new_batch[param_name] = deepcopy(torch.from_numpy(batch[param_name]))
        else:
            new_batch[param_name] = np.zeros((new_num_features, ))
            for i in range(new_num_features):
                new_batch[param_name][i] = batch[param_name][i]
            new_batch[param_name] = torch.from_numpy(new_batch[param_name])
    return new_batch

def widen_child_linear_helper(linear: OrderedDict, mapping, noise_factor=5e-2):
    """
    linear layer only have weight and bias two parameters
    """
    new_linear = collections.OrderedDict()
    linear['weight'] = linear['weight'].numpy()
    out_features = linear['weight'].shape[0]
    new_linear['weight'] = np.zeros((out_features, len(mapping)))
    if 'bias' in linear.keys():
        new_linear['bias'] = deepcopy(linear['bias'])
    scale = [mapping.count(m) for m in mapping]
    for i in range(len(mapping)):
        new_linear['weight'][:, i] = \
            linear['weight'][:, mapping[i]].copy() / scale[i]
    new_linear['weight'] = torch.from_numpy(new_linear['weight'])
    return new_linear

def widen_parent_linear_helper(linear: OrderedDict, mapping, noise_factor=5e-2):
    new_linear = collections.OrderedDict()
    linear['weight'] = linear['weight'].numpy()
    if 'bias' in linear.keys():
        linear['bias'] = linear['bias'].numpy()
        new_linear['bias'] = np.zeros((len(mapping),))
    in_features = linear['weight'].shape[1]
    new_linear['weight'] = np.zeros((len(mapping), in_features))

    for i in range(len(mapping)):
        new_linear['weight'][i, :] =\
            linear['weight'][mapping[i], :].copy() + np.random.normal(scale=noise_factor, size=list(linear['weight'][mapping[i], :].shape))
        if 'bias' in linear.keys():
            new_linear['bias'][i] =\
                linear['bias'][mapping[i]].copy()
    new_linear['weight'] = torch.from_numpy(new_linear['weight'])
    new_linear['bias'] = torch.from_numpy(new_linear['bias'])
    return new_linear

def shrink_linear_helper(linear: OrderedDict, new_in_features, new_out_features, noise_factor=5e-2):
    new_linear = collections.OrderedDict()
    linear['weight'] = linear['weight'].numpy()
    if 'bias' in linear.keys():
        linear['bias'] = linear['bias'].numpy()
        new_linear['bias'] = np.zeros((new_out_features, ))
    new_linear['weight'] = np.zeros((new_out_features, new_in_features))

    for i in range(new_out_features):
        for j in range(new_in_features):
            new_linear['weight'][i, j] = linear['weight'][i,j]
        if 'bias' in linear.keys():
            new_linear['bias'][i] = linear['bias'][i]
    new_linear['weight'] = torch.from_numpy(new_linear['weight'])
    new_linear['bias'] = torch.from_numpy(new_linear['bias'])
    return new_linear

def widen_parent_conv(torch_model, layer_name, ratio: int=2, noise_factor=5e-2):
    old_layer = get_model_layer(torch_model, layer_name)
    old_param = old_layer.state_dict()
    mapping = list(range(old_layer.out_channels)) * ratio
    new_param = widen_parnet_conv_helper(old_param, mapping, noise_factor=noise_factor)
    new_layer = torch.nn.Conv2d(
        old_layer.in_channels,
        len(mapping),
        old_layer.kernel_size,
        stride=old_layer.stride,
        padding=old_layer.padding,
        groups=old_layer.groups,
        bias=True if old_layer.bias is not None else False
    )
    new_layer.load_state_dict(new_param)
    set_model_layer(torch_model, new_layer, layer_name)
    return torch_model

def widen_child_conv(torch_model, layer_name, ratio: int=2, noise_factor=5e-2):
    old_layer = get_model_layer(torch_model, layer_name)
    groups = old_layer.groups
    mapping = list(range(old_layer.in_channels // groups)) * ratio
    old_param = old_layer.state_dict()
    new_param = widen_child_conv_helper(old_param, mapping, noise_factor=noise_factor)
    new_layer = torch.nn.Conv2d(
        len(mapping) * old_layer.groups,
        old_layer.out_channels,
        old_layer.kernel_size,
        stride=old_layer.stride,
        padding=old_layer.padding,
        groups=old_layer.groups,
        bias=True if old_layer.bias is not None else False
    )
    new_layer.load_state_dict(new_param)
    set_model_layer(torch_model, new_layer, layer_name)
    return torch_model

def shrink_conv(torch_model, layer_name, ratio: float=0.5, is_first: bool=False, is_last: bool=False, noise_factor=5e-2):
    old_layer = get_model_layer(torch_model, layer_name)
    groups = old_layer.groups
    in_channel = old_layer.in_channels
    out_channel = old_layer.out_channels
    new_in_channel = max(int(in_channel * ratio), 1) if not is_first else in_channel
    new_out_channel = max(int(out_channel * ratio), 1) if not is_last else out_channel
    new_group = 1
    for i in range(groups, 0, -1):
        if new_in_channel % i == 0 and new_out_channel % i == 0:
            new_group = i
            break
    old_param = old_layer.state_dict()
    # logging.info(f"new_in_channel: {new_in_channel}, new_out_channel: {new_out_channel}, new_groups: {new_group}")
    # logging.info(f"in_channel: {in_channel}, out_channel: {out_channel}, groups: {groups}")
    in_weights = new_in_channel // new_group
    out_weights = new_out_channel
    # logging.info(f"in_weights: {in_weights}, out_weights: {out_weights}")
    new_param = shrink_conv_helper(old_param, new_in_channel=in_weights, new_out_channel=out_weights)
    # logging.info(f"new_weights: {new_param['weight'].shape}, weights: {old_param['weight'].shape}")
    new_layer = torch.nn.Conv2d(
        new_in_channel,
        new_out_channel,
        old_layer.kernel_size,
        stride=old_layer.stride,
        padding=old_layer.padding,
        groups=new_group,
        bias=True if old_layer.bias is not None else False
    )
    # logging.info(f"new layer weights: {new_layer.state_dict()['weight'].shape}")
    # logging.info(f"new layer: {new_layer}")
    new_layer.load_state_dict(new_param)
    set_model_layer(torch_model, new_layer, layer_name)
    return torch_model

def widen_bn(torch_model, layer_name, ratio: int=2, noise_factor=5e-2):
    old_layer = get_model_layer(torch_model, layer_name)
    old_param = old_layer.state_dict()
    mapping = list(range(old_layer.num_features)) * ratio
    new_param = widen_batch_helper(old_param, mapping, noise_factor=noise_factor)
    new_layer = torch.nn.BatchNorm2d(
        num_features=len(mapping),
        eps=old_layer.eps,
        momentum=old_layer.momentum,
        affine=old_layer.affine,
        track_running_stats=old_layer.track_running_stats
    )
    new_layer.load_state_dict(new_param)
    set_model_layer(torch_model, new_layer, layer_name)
    return torch_model

def shrink_bn(torch_model, layer_name, ratio: float=0.5, noise_factor=5e-2):
    old_layer = get_model_layer(torch_model, layer_name)
    old_param = old_layer.state_dict()
    num_features = max(int(old_layer.num_features * ratio), 1)
    new_param = shrink_batch_helper(old_param, new_num_features=num_features)
    new_layer = torch.nn.BatchNorm2d(
        num_features=num_features,
        eps=old_layer.eps,
        momentum=old_layer.momentum,
        affine=old_layer.affine,
        track_running_stats=old_layer.track_running_stats
    )
    new_layer.load_state_dict(new_param)
    set_model_layer(torch_model, new_layer, layer_name)
    return torch_model

def widen_child_ln(torch_model, layer_name, ratio: int=2, noise_factor=5e-2):
    old_layer = get_model_layer(torch_model, layer_name)
    old_param = old_layer.state_dict()
    mapping = list(range(old_layer.in_features)) * ratio
    new_param = widen_child_linear_helper(old_param, mapping, noise_factor)
    new_layer = torch.nn.Linear(
        len(mapping),
        old_layer.out_features,
        bias = True if old_layer.bias is not None else False
    )
    new_layer.load_state_dict(new_param)
    set_model_layer(torch_model, new_layer, layer_name)
    return torch_model

def widen_parent_ln(torch_model, layer_name, ratio: int=2, noise_factor=5e-2):
    old_layer = get_model_layer(torch_model, layer_name)
    old_param = old_layer.state_dict()
    mapping = list(range(old_layer.out_features)) * ratio
    new_param = widen_parent_linear_helper(old_param, mapping, noise_factor)
    new_layer = torch.nn.Linear(
        old_layer.in_features,
        len(mapping),
        bias = True if old_layer.bias is not None else False
    )
    new_layer.load_state_dict(new_param)
    set_model_layer(torch_model, new_layer, layer_name)
    return torch_model

def shrink_ln(torch_model, layer_name, ratio: float=0.5, is_last: bool=False, noise_factor=5e-2):
    old_layer = get_model_layer(torch_model, layer_name)
    old_param = old_layer.state_dict()
    new_in_features = max(int(old_layer.in_features * ratio), 1)
    new_out_features = old_layer.out_features
    if not is_last:
        new_out_features = max(int(old_layer.out_features * ratio), 1)
    new_param = shrink_linear_helper(old_param, new_in_features, new_out_features)
    new_layer = torch.nn.Linear(
        new_in_features,
        new_out_features,
        bias = True if old_layer.bias is not None else False
    )
    new_layer.load_state_dict(new_param)
    set_model_layer(torch_model, new_layer, layer_name)
    return torch_model
    
def deepen(torch_model, layer_name):
    old_layer = deepcopy(get_model_layer(torch_model, layer_name))
    in_channels = old_layer.out_channels
    kernel_size = old_layer.kernel_size[0]
    if (kernel_size - 1) % 2 == 0:
        padding = (kernel_size - 1) // 2
    else:
        padding = 'same'
    new_conv = torch.nn.Conv2d(
        in_channels, in_channels, kernel_size,
        padding=padding, bias=False
    )
    new_conv_param = new_conv.state_dict()
    new_conv_param['weight'] = torch.nn.init.dirac_(new_conv_param['weight'])
    new_conv.load_state_dict(new_conv_param)
    new_bn = torch.nn.BatchNorm2d(in_channels)
    new_layer = torch.nn.Sequential(
        old_layer,
        new_bn,
        new_conv
    )
    set_model_layer(torch_model, new_layer, layer_name)
    return torch_model


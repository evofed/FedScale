from fedscale.core.logger.aggragation import logDir

import sys
import pickle
import os
import torch
import math
from copy import deepcopy
from typing import List, Iterable, Tuple
import numpy as np
from collections import defaultdict, OrderedDict
import logging

from fedscale.core.net2netlib import get_model_layer, set_model_layer
from fedscale.core.model_layers import *
from fedscale.core.fluid_lib import get_neuron_weight_diff


def reset_parameters(model: torch.nn.Module):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        else:
            reset_parameters(layer)

class Fluid_Model_Manager:
    def __init__(self, model, args) -> None:
        reset_parameters(model)
        self.model = model
        self.last_gradient_weights = None
        self.model_weights = OrderedDict()
        self.models_to_test = {}
        self.client_models = {}
        self.client2p = {}
        self.layer_diffs = []
        self.th = {}
        self.invariant_neurons = {}
        self.dropped_neurons = defaultdict(list)
        self.def_invariant_neurons = {}
        self.task_this_round = 0
        self.task_updated = 0
        if args.data_set == "femnist":
            self.model_layers = femnist_model_layers
            self.th_incre = femnist_th_incre
        else:
            raise NotImplementedError(f"Dataset {args.data_set} is not supported")
        
        self.layer_inout_dims = {}
        for layer_name in self.model_layers.keys():
            layer = get_model_layer(self.model, layer_name)
            if isinstance(layer, torch.nn.Conv2d):
                self.layer_inout_dims[layer_name] = (layer.in_channels, layer.out_channels)
            elif isinstance(layer, torch.nn.Linear):
                self.layer_inout_dims[layer_name] = (layer.in_features, layer.out_features)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                self.layer_inout_dims[layer_name] = (layer.num_features, layer.num_features)
            else:
                raise NotImplementedError(f"Layer {layer_name} is not supported")
    
    def get_model_update_size(self):
        return sys.getsizeof(pickle.dumps(self.model)) // 1024.0 * 8.0
    
    def _expand_sub_model(self, sub_model, layer_drop_in, layer_drop_out):
        for layer_name in self.model_layers.keys():
            layer = get_model_layer(sub_model, layer_name)
            if isinstance(layer, torch.nn.Conv2d):
                num_in_filter = layer.in_channels
                num_out_filter = layer.out_channels
                weight = layer.weight.detach().numpy()
                for out_id in layer_drop_out[layer_name]:
                    weight = np.insert(weight, out_id, 0, axis=0)
                for in_id in layer_drop_in[layer_name]:
                    weight = np.insert(weight, in_id, 0, axis=1)
                assert weight.shape[0] == num_out_filter + len(layer_drop_out[layer_name])
                assert weight.shape[1] == num_in_filter + len(layer_drop_in[layer_name])
                new_layer = torch.nn.Conv2d(weight.shape[1], weight.shape[0], 
                                            kernel_size=layer.kernel_size, 
                                            stride=layer.stride, 
                                            padding=layer.padding, 
                                            dilation=layer.dilation, 
                                            groups=layer.groups, 
                                            bias=layer.bias is not None)
                new_layer.weight = torch.nn.Parameter(torch.tensor(weight))
                set_model_layer(sub_model, new_layer, layer_name)
            elif isinstance(layer, torch.nn.Linear):
                num_in_neurons = layer.in_features
                num_out_neurons = layer.out_features
                weight = layer.weight.detach().numpy()
                for out_id in layer_drop_out[layer_name]:
                    weight = np.insert(weight, out_id, 0, axis=0)
                for in_id in layer_drop_in[layer_name]:
                    weight = np.insert(weight, in_id, 0, axis=1)
                assert weight.shape[0] == num_out_neurons + len(layer_drop_out[layer_name])
                assert weight.shape[1] == num_in_neurons + len(layer_drop_in[layer_name])
                new_layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=layer.bias is not None)
                new_layer.weight = torch.nn.Parameter(torch.tensor(weight))
                set_model_layer(sub_model, new_layer, layer_name)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                num_features = layer.num_features
                weight = layer.weight.detach().numpy()
                for id in layer_drop_out[layer_name]:
                    weight = np.insert(weight, id, 0, axis=0)
                assert weight.shape[0] == num_features + len(layer_drop_out[layer_name])
                new_layer = torch.nn.BatchNorm2d(weight.shape[0], 
                                                eps=layer.eps, 
                                                momentum=layer.momentum, 
                                                affine=layer.affine, 
                                                track_running_stats=layer.track_running_stats)
                new_layer.weight = torch.nn.Parameter(torch.tensor(weight))
                set_model_layer(sub_model, new_layer, layer_name)
            else:
                raise NotImplementedError(f"Layer {layer_name} is not supported")
        return sub_model
    
    def weight_aggregation(self, results):
        # content of results
        # 'clientId': id of the client  int
        # 'cap': capacity of the client float
        # 'moving_loss': moving loss of the client  float
        # 'update_weight': update weight of the client  dict (same key as the ordered dict of the model)
        self.task_updated += 1
        clientId = results['clientId']
        # save layer diffs for updating th
        client_model = self.client_models[clientId].load_state_dict(results['update_weight'])
        if self.client2p[clientId] == 1:
            self.layer_diffs.append(get_neuron_weight_diff(self.model, client_model, self.model_layers.keys()))
        else:
            # expand the sub model if p < 1
            client_model = self._expand_sub_model(client_model, self.client_drop_ins[clientId], self.client_drop_outs[clientId])
        
        # normal aggregation
        if self.task_updated == 1:
            self.model_weights = deepcopy(client_model.state_dict())
        else:
            for name, param in client_model.state_dict().items():
                self.model_weights[name] += param

        if self.task_updated == self.task_this_round:
            for name in self.model_weights.keys():
                self.model_weights[name] /= self.task_this_round

    def init_th(self):
        self.th = {}
        for layer in self.model_layers.keys():
            self.th[layer] = np.mean([np.mean(lydf[layer]) for lydf in self.layer_diffs])
        logging.info(f"Initial thresholds: {self.th}")
    
    def identify_invariant_neurons(self):
        # if 75% of the clients' neurons are less than the threshold, then the neuron is invariant
        self.invariant_neurons = defaultdict(list)
        for layer in self.model_layers.keys():
            for neuron in range(len(self.layer_diffs[0][layer])):
                count = 0
                for lydf in self.layer_diffs:
                    if lydf[layer][neuron] < self.th[layer]:
                        count += 1
                if count / len(self.layer_diffs) > 0.75:
                    self.invariant_neurons[layer].append(neuron)
        # if a dropped neuron is invariant, then it's strongly invariant
        self.def_invariant_neurons = defaultdict(list)
        for layer in self.model_layers.keys():
            for neuron in self.dropped_neurons[layer]:
                if neuron in self.invariant_neurons[layer]:
                    self.def_invariant_neurons[layer].append(neuron)

    def calibrate(self, round):
        client_cap = sorted(self.client2p.items(), key=lambda x: x[1])
        slowest_client = client_cap[0][1]
        next_slowest_client = client_cap[1][1]
        speed_up = next_slowest_client / slowest_client
        if speed_up >= 0.9:
            self.p = 0.95
        elif speed_up >= 0.8:
            self.p = 0.85
        elif speed_up >= 0.7:
            self.p = 0.75
        elif speed_up >= 0.6:
            self.p = 0.65
        else:
            self.p = 0.5

        logging.info(f"update self.p to {self.p} for round {round}.")

        self.models_to_test[self.p] = None

        if round >= 10:
            self.th = {layer: th + self.th_incre for layer, th in self.th.items()}
            logging.info(f"update thresholds to {self.th} for round {round}.")

    def save_last_param(self):
        self.last_model = deepcopy(self.model)

    def load_model_weight(self):
        # load model_weights to model
        self.model.load_state_dict(self.model_weights)

    def save_model(self):
        model_path = os.path.join(logDir, 'model.pth.tar')
        with open(model_path, 'wb') as model_out:
            pickle.dump(self.model, model_out)

    def get_model_mac_by_p(self, p) -> float:
        # return the MAC of the model of p
        return .0
    
    def get_model_mac_by_client(self, client_id) -> float:
        # return the MAC of the model of client
        return .0

    def get_model(self, client_id):
        assert client_id in self.client_models
        return self.client_models[client_id]

    def get_clients(self) -> list:
        return list(self.client_models.keys())
    
    def _get_dropping_neurons(self, p: float) -> Tuple[dict, dict]:
        # return the neurons to drop
        layer_drop_out = {}
        for layer_name in self.model_layers.keys():
            layer_drop_out[layer_name] = []
            if self.model_layers[layer_name]["descendants"] == []:
                continue
            num_drop = math.floor(self.layer_inout_dims[layer_name][1] * (1-p))
            if num_drop <= self.def_invariant_neurons[layer_name]:
                layer_drop_out[layer_name] = self.def_invariant_neurons[layer_name][:num_drop]
            elif num_drop <= self.invariant_neurons[layer_name]:
                layer_drop_out[layer_name] = self.def_invariant_neurons[layer_name][:num_drop]
                possible_invariant_neurons = [neuron for neuron in self.invariant_neurons[layer_name] \
                                              if neuron not in self.def_invariant_neurons[layer_name]]
                layer_drop_out[layer_name] += np.random.choice(possible_invariant_neurons,
                                                                    num_drop - len(self.def_invariant_neurons[layer_name]),
                                                                    replace=False).tolist()
            else:
                layer_drop_out[layer_name] = self.invariant_neurons[layer_name]
                variant_neurons = [neuron for neuron in range(self.layer_inout_dims[layer_name][1]) \
                                   if neuron not in self.invariant_neurons[layer_name]]
                layer_drop_out[layer_name] += np.random.choice(variant_neurons, 
                                                                    num_drop - len(self.invariant_neurons[layer_name]), 
                                                                    replace=False).tolist()
            layer_drop_out[layer_name] = sorted(layer_drop_out[layer_name])
        layer_drop_in = {}
        for layer_name in self.model_layers.keys():
            if self.model_layers[layer_name]["ansestors"] == []:
                layer_drop_in[layer_name] = []
            layer_drop_in[layer_name] = layer_drop_out[self.model_layers[layer_name]["ansestors"][0]]

        return layer_drop_in, layer_drop_out
    
    def generate_sub_model(self, p: float):
        if p == 1:
            return deepcopy(self.model)
        layer_drop_in, layer_drop_out = self._get_dropping_neurons(p)
        sub_model = deepcopy(self.model)
        for layer_name in self.model_layers.keys():
            layer = get_model_layer(sub_model, layer_name)
            if isinstance(layer, torch.nn.Conv2d):
                num_in_filter = layer.in_channels
                num_out_filter = layer.out_channels
                new_num_in_filter = math.ceil(num_in_filter * p) if len(self.model_layers[layer_name]["ansestors"]) != 0 else num_in_filter
                new_num_out_filter = math.ceil(num_out_filter * p) if len(self.model_layers[layer_name]["descendants"]) != 0 else num_out_filter
                new_in_filter_ids = [id for id in list(range(num_in_filter)) if id not in layer_drop_in[layer_name]]
                new_out_filter_ids = [id for id in list(range(num_out_filter)) if id not in layer_drop_out[layer_name]]
                assert len(new_in_filter_ids) == new_num_in_filter
                assert len(new_out_filter_ids) == new_num_out_filter
                new_weight = layer.weight[new_out_filter_ids][:, new_in_filter_ids]
                new_layer = torch.nn.Conv2d(new_num_in_filter, new_num_out_filter, 
                                           kernel_size=layer.kernel_size, 
                                           stride=layer.stride, 
                                           padding=layer.padding, 
                                           dilation=layer.dilation, 
                                           groups=layer.groups, 
                                           bias=layer.bias is not None)
                new_layer.weight = torch.nn.Parameter(new_weight)
                set_model_layer(sub_model, new_layer, layer_name)
            elif isinstance(layer, torch.nn.Linear):
                num_in_feature = layer.in_features
                num_out_feature = layer.out_features
                new_num_in_feature = math.ceil(num_in_feature * p) if len(self.model_layers[layer_name]["ansestors"]) != 0 else num_in_feature
                new_num_out_feature = math.ceil(num_out_feature * p) if len(self.model_layers[layer_name]["descendants"]) != 0 else num_out_feature
                new_in_feature_ids = [id for id in list(range(num_in_feature)) if id not in layer_drop_in[layer_name]]
                new_out_feature_ids = [id for id in list(range(num_out_feature)) if id not in layer_drop_out[layer_name]]
                assert len(new_in_feature_ids) == new_num_in_feature
                assert len(new_out_feature_ids) == new_num_out_feature
                new_weight = layer.weight[new_out_feature_ids][:, new_in_feature_ids]
                new_layer = torch.nn.Linear(new_num_in_feature, new_num_out_feature, bias=layer.bias is not None)
                new_layer.weight = torch.nn.Parameter(new_weight)
                set_model_layer(sub_model, new_layer, layer_name)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                num_features = layer.num_features
                new_num_features = math.ceil(num_features * p)
                new_feature_ids = [id for id in list(range(num_features)) if id not in layer_drop_out[layer_name]]
                assert len(new_feature_ids) == new_num_features
                new_weight = layer.weight[new_feature_ids]
                new_layer = torch.nn.BatchNorm2d(new_num_features, 
                                                eps=layer.eps, 
                                                momentum=layer.momentum, 
                                                affine=layer.affine, 
                                                track_running_stats=layer.track_running_stats)
                new_layer.weight = torch.nn.Parameter(new_weight)
                set_model_layer(sub_model, new_layer, layer_name)
            else:
                raise NotImplementedError(f"Layer {layer_name} is not supported")
        return sub_model, layer_drop_in, layer_drop_out

    def register_tasks(self, clients: list, caps: dict, is_first_round: bool=False):
        # clean up the assignment from last round
        self.client2p = {}
        self.client_models = {}
        self.client_drop_ins = {}
        self.client_drop_outs = {}
        self.dropped_neurons = defaultdict(list)
        # assign clients to models
        self.client_caps = caps
        self.task_this_round = len(clients)
        self.task_updated = 0
        for client in clients:
            if is_first_round:
                self.client2p[client] = 1
                self.client_models[client] = deepcopy(self.model)
                logging.info(f"using full model for client {client}")
                continue
            cap = caps[client]
            self.client2p[client] = 1 if cap > femnist_target_mac else self.p
            self.client_models[client], self.client_drop_ins[client], self.client_drop_outs[client] = \
                self.generate_sub_model(self.client2p[client])
            logging.info(f"using sub model for client {client} with p={self.client2p[client]}, dropping neurons: {self.client_drop_outs[client]}")
            # add dropped neurons
            for layer_name in self.client_drop_outs:
                self.dropped_neurons[layer_name] += self.client_drop_outs[layer_name]
        # clean up dropped neurons
        for layer_name in self.dropped_neurons:
            self.dropped_neurons[layer_name] = list(set(self.dropped_neurons[layer_name]))

        logging.info(f"all dropped neurons this round: {self.dropped_neurons}")

    def prepare_to_test(self) -> list:
        for p in self.models_to_test.keys():
            self.models_to_test[p], _, _ = self.generate_sub_model(p)
        return list(self.models_to_test.keys())
    
    def get_model_by_p(self, p):
        assert p in self.models_to_test and self.models_to_test[p] is not None
        return self.models_to_test[p]

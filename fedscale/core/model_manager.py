import logging
import torch
import networkx, onnx
from onnx.helper import printable_graph
from collections import defaultdict
from fedscale.core.net2netlib import *
from copy import deepcopy
import random
import numpy as np
import math

omit_operator = ['Identity', 'Constant']
conflict_operator = ['Add']
weight_operator = ['Conv', 'Gemm']

class Widen_Operator():
    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

    def update_ratio(self, ratio) -> None:
        self.ratio = ratio
    
    def get_distance(self) -> float:
        return self.ratio

class Deepen_Operator():
    def __init__(self, num: int, kernel_size: list, in_channel: list, out_channel: list) -> None:
        self.num = num
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

    def get_distance(self):
        return self.num

class ONNX_Edge():
    def __init__(self, tensor_inputs: list, outputs: list, param_inputs: list, operator: str) -> None:
        self.tensor_inputs = tensor_inputs
        self.outputs = outputs
        self.param_inputs = param_inputs
        self.operator = operator
        # if the edge has parameter input, then use the parameter as the name
        if len(param_inputs) > 0:
            param_input = param_inputs[0]
            if 'bias' in param_input or 'weight' in param_input:
                param_input = param_input.split('.')[:-1]
                param_input = '.'.join(param_input)
                self.name = param_input
            else:
                raise Exception(f'no bias or weight detected in parameter input')
        else:
            self.name = self.operator
        


class Model_Manager():
    def __init__(self, torch_model, seed=2333, candidate_capacity=5) -> None:
        self.seed = seed
        self.base_model = torch_model
        self.base_graph = []
        self.widen_trajectary = []
        self.deepen_trajectary = []
        self.base_dag = None
        self.name2id = {} 
        self.candidate_models = [deepcopy(torch_model) for _ in range(candidate_capacity)]

    
    def translate_base_model(self):
        # 1. translate self.base_model to temporary onnx model
        # 2. parse onnx model to directed acyclic diagram

        dummy_input = torch.randn(10, 3, 224, 224)
        torch.onnx.export(self.base_model, dummy_input, 'tmp.onnx',
            export_params=True, verbose=0, training=1, do_constant_folding=False)
        onnx_model = onnx.load('tmp.onnx')
        graph = onnx_model.graph
        graph_string = printable_graph(graph)
        start, end = graph_string.find('{'), graph_string.find('}')
        graph_string = graph_string[start+2:end]
        edge_list = graph_string.split('\n')
        input2id = defaultdict(list)
        self.base_dag = networkx.DiGraph()
        
        # construct nodes
        for edge_string in edge_list:
            equal_pos = edge_string.find('=')
            if equal_pos == -1:
                continue
            left, right = edge_string[:equal_pos], edge_string[equal_pos+1:]
            # extract output nodes
            outputs = left.strip()[1:].split(',')
            outputs = [output.strip() for output in outputs]
            
            # extract operators
            left_p = right.find('(')
            func_sig = right[:left_p].strip()
            left_pt = func_sig.find('[')
            if left_pt == -1:
                operator = func_sig[:left_p]
            else:
                operator = func_sig[:left_pt]
            if operator in omit_operator:
                continue

            # extract all input nodes
            left_p = right.find('(')
            func_body = right[left_p+1:].strip()[:-1]
            inputs = func_body.split(',')
            inputs = [node.strip()[1:] for node in inputs]
            tensor_inputs = []
            param_inputs = []
            for node in inputs:
                if 'bias' in node or 'weight' in node:
                    param_inputs.append(node)
                else:
                    tensor_inputs.append(node)
            
            # construct onnx node and networkx node
            node = ONNX_Edge(tensor_inputs, outputs, param_inputs, operator)
            node_id = len(self.base_dag.nodes())
            self.name2id[node.name] = node_id
            self.base_dag.add_node(node_id, attr=node)
            for tensor_input in node.tensor_inputs:
                input2id[tensor_input].append(node_id)
        
        # construct edges
        for node_id in self.base_dag.nodes():
            outputs = self.base_dag.nodes()[node_id]['attr'].outputs
            for output in outputs:
                for next_id in input2id[output]:
                    self.base_dag.add_edge(node_id, next_id)
        
    def get_all_nodes(self):
        if self.base_dag == None:
            self.translate_base_model()
        nodes = []
        for node_id in self.base_dag.nodes():
            nodes.append(self.base_dag.nodes()[node_id]['attr'])
        return nodes

    def get_all_edges(self):
        if self.base_dag == None:
            self.translate_base_model()
        return self.base_dag.edges()

    def get_base_convs(self):
        convs = []
        for node_id in self.base_dag.nodes():
            if 'Conv' in self.base_dag.nodes()[node_id]['attr'].operator:
                convs.append([node_id, self.base_dag.nodes()[node_id]['attr'].name])
        return convs

    def get_weighted_layers(self):
        layers = []
        for node_id in self.base_dag.nodes():
            if self.base_dag.nodes()[node_id]['attr'].operator in weight_operator:
                layers.append([node_id, self.base_dag.nodes()[node_id]['attr'].name])
        return layers

    def get_base_parents(self, query_node_id):
        # current only support resnet, mobilenet_v2, alexnet, regnet_x_16gf, vgg19_bn
        # not support shufflenet
        # other nets are not tested yet
        l = []
        for node_id in self.base_dag.predecessors(query_node_id):
            node = self.base_dag.nodes()[node_id]['attr']
            if node.operator == 'BatchNormalization':
                l.append(node_id)
            if node.operator in weight_operator:
                l.append(node_id)
            else:
                l += self.get_base_parents(node_id)
        return l

    def get_base_children(self, query_node_id):
        # current only support resnet, mobilenet_v2, alexnet, regnet_x_16gf, vgg19_bn
        # not support shufflenet
        # other nets are not tested yet
        l = []
        for node_id in self.base_dag.successors(query_node_id):
            node = self.base_dag.nodes()[node_id]['attr']
            if node.operator == 'BatchNormalization':
                l.append(node_id)
            if node.operator in weight_operator:
                l.append(node_id)
            else:
                l += self.get_base_children(node_id)
        return l

    def is_conflict(self, query_node_id):
        conflict_id = -1
        for node_id in self.base_dag.successors(query_node_id):
            node = self.base_dag.nodes()[node_id]['attr']
            if node.operator in conflict_operator:
                return node_id
            elif node.operator not in weight_operator:
                temp_result = self.is_conflict(node_id)
                if temp_result != -1:
                    return temp_result
        return conflict_id
    
    def get_add_operand(self, add_node_id):
        return self.get_base_parents(add_node_id)
    
    def get_base_neighbour(self, query_node_id):
        # current only support resnet, mobilenet_v2, alexnet, regnet_x_16gf, vgg19_bn
        # not support shufflenet
        # other nets are not tested yet
        l = []
        conflict_node_id = self.is_conflict(query_node_id)
        while conflict_node_id != -1:
            l += self.get_add_operand(conflict_node_id)
            conflict_node_id = self.is_conflict(conflict_node_id)
        return l

    def get_base_widen_instruction(self, query_node_id):
        child_convs = set()
        parent_convs = set()
        child_lns = set()
        parent_lns = set()
        bns = set()
        
        children = self.get_base_children(query_node_id)
        neighbours = self.get_base_neighbour(query_node_id)
        for neighbor in neighbours:
            node = self.base_dag.nodes()[neighbor]['attr']
            if node.operator == 'BatchNormalization':
                bns.add(neighbor)
            elif node.operator == 'Gemm':
                parent_lns.add(neighbor)
            else:
                parent_convs.add(neighbor)
            children += self.get_base_children(neighbor)
        for child in children:
            node = self.base_dag.nodes()[child]['attr']
            if node.operator == 'BatchNormalization':
                bns.add(child)
            elif node.operator == 'Gemm':
                child_lns.add(child)
            else:
                child_convs.add(child)
        return list(child_convs), list(parent_convs), list(bns), list(child_lns), list(parent_lns)
    
    def get_base_deepen_instruction(self, query_node_id):
        out_channel, in_channel = None, None

        # get out_channels
        child = self.get_base_children(query_node_id)[0]
        child_node = self.base_dag.nodes()[child]['attr']
        if child_node.operator == 'BatchNormalization':
            bn = get_model_layer(self.base_model, child_node.name)
            out_channel = bn.num_features
        else:
            conv = get_model_layer(self.base_model, child_node.name)
            out_channel = conv.in_channels
        
        # get out_channels
        parent_node = self.base_dag.nodes()[parent_node]['attr']
        conv = get_model_layer(self.base_model, parent_node.name)
        in_channel = conv.out_channels

        return in_channel, out_channel

    def widen_layer(self, candidate_id, node_id):
        node_name = self.base_dag.nodes()[node_id]['attr'].name
        children, parents, bns, ln_children, ln_parents = self.get_base_widen_instruction(node_id)
        node = self.base_dag.nodes()[node_id]['attr']
        if len(children) == 0 and len(ln_children) == 0:
            print(f'fail to widen {node.name} as it is the last layer')
            return []
        if node.operator == 'Gemm':
            ln_parents.append(node_id)
        elif node_id not in parents:
            parents.append(node_id)
        for child in children:
            node_name = self.base_dag.nodes()[child]['attr'].name
            self.candidate_models[candidate_id] = widen_child_conv(
                self.candidate_models[candidate_id], node_name)
        for parent in parents:
            node_name = self.base_dag.nodes()[parent]['attr'].name
            self.candidate_models[candidate_id] = widen_parent_conv(
                self.candidate_models[candidate_id], node_name)
        for bn in bns:
            node_name = self.base_dag.nodes()[bn]['attr'].name
            self.candidate_models[candidate_id] = widen_bn(
                self.candidate_models[candidate_id], node_name)
        for ln_child in ln_children:
            node_name = self.base_dag.nodes()[ln_child]['attr'].name
            self.candidate_models[candidate_id] = widen_child_ln(
                self.candidate_models[candidate_id], node_name
            )
        for ln_parent in ln_parents:
            node_name = self.base_dag.nodes()[ln_child]['attr'].name
            node_name = self.base_dag.nodes()[ln_parent]['attr'].name
            self.candidate_models[candidate_id] = widen_parent_ln(
                self.candidate_models[candidate_id], node_name
            )
        return parents

    def deepen_layer(self, candidate_id, node_id):
        node = self.base_dag.nodes()[node_id]['attr']
        if node.operator == 'Conv':
            self.candidate_models[candidate_id] = deepen(
                self.candidate_models[candidate_id], node.name
            )

    def base_model_scale(self, alpha: float=1.32, beta: float=1.21):
        """
        EfficientNet style model scaling
        d = alpha^phi
        w = beta^phi
        scale the base model to num super models
        """
        layers = self.get_weighted_layers()
        random.shuffle(layers)
        for candidate_id in range(len(self.candidate_models)):
            # widen
            widen_p = alpha - 1
            widen_num = 0
            widened_layers = []
            for node_id, _ in layers:
                dice = random.uniform(0,1)
                if dice < widen_p:
                    widened_layers += self.widen_layer(candidate_id, node_id)
                    widen_num += len(widened_layers)
                if widen_num > widen_p * len(layers):
                    break
            # deepen
            deepen_p = beta - 1
            deepened_layers = []
            for node_id, _ in layers:
                dice = random.uniform(0,1)
                if dice < deepen_p:
                    self.deepen_layer(candidate_id, node_id)
                    deepened_layers.append(node_id)
            # record trajectary
            layers = self.get_weighted_layers()
            layers_onehot = np.array([0 for _ in range(len(layers))])
            def nodeid2convid(node_id):
                for i, conv in enumerate(layers):
                    if conv[0] == node_id:
                        return i
                raise Exception(f"not find node {node_id}")
            # record widen trajectary
            if len(self.widen_trajectary) == candidate_id:
                self.widen_trajectary.append(layers_onehot)
            else:
                self.widen_trajectary[candidate_id] = layers_onehot
            for node_id in widened_layers:
                conv_id = nodeid2convid(node_id)
                self.widen_trajectary[candidate_id][conv_id] = 1
            # record deepen trajectary
            if len(self.deepen_trajectary) == candidate_id:
                self.deepen_trajectary.append(layers_onehot)
            else:
                self.deepen_trajectary[candidate_id] = layers_onehot
            for node_id in deepened_layers:
                conv_id = nodeid2convid(node_id)
                self.deepen_trajectary[candidate_id][conv_id] = 1
    
    def get_candidate_distance(self, candidate_i, candidate_j):
        """
        calculate the model distance based on widen and deepen trajectary
        distance <- [0, +inf]
        """
        widen_i = self.widen_trajectary[candidate_i]
        deepen_i = self.deepen_trajectary[candidate_i]
        widen_j = self.widen_trajectary[candidate_j]
        deepen_j = self.deepen_trajectary[candidate_j]
        return np.linalg.norm(widen_i - widen_j) + np.linalg.norm(deepen_i - deepen_j)
    
    def get_candidate_similarity(self, candidate_i, candidate_j):
        """
        calculate the similarity of two models based on model distance:
        similarity = 2 - 2 * sigmoid(distance)
        similarity <- [0,1]
        """
        distance = self.get_candidate_distance(candidate_i, candidate_j)
        return 2.0 - 2.0 / (1.0 + math.exp(-distance))

    def reset_base(self, candidate_id, candidate_capacity: int=5):
        self.base_model = deepcopy(self.candidate_models[candidate_id])
        self.candidate_models = [deepcopy(self.base_model) for _ in range(candidate_capacity)]
        self.base_graph = []
        self.widen_trajectary = []
        self.deepen_trajectary = []
        self.base_dag = None
        self.name2id = {} 
        self.translate_base_model()

    def aggregate_weights(self, weights: OrderedDict, comming_weights: dict, model_id, comming_model_id, device, is_init: bool=False):
        """
        softly add weight from comming model to model
        return aggregated weights and aggregation status
        """
        if model_id != comming_model_id:
            similarity = self.get_candidate_similarity(model_id, comming_model_id)
            for param in weights.keys():
                if param in comming_weights.keys():
                    # change data type
                    param_weight = comming_weights[param]
                    if isinstance(param_weight, list):
                        param_weight = np.asarray(param_weight, dtype=np.float32)
                    param_weight = torch.from_numpy(param_weight).to(device=device)
                    # check data type
                    if weights[param].data.dtype == torch.int64:
                        if is_init:
                            weights[param].data = param_weight
                        continue
                    if len(comming_weights[param].shape) != len(weights[param].shape):
                        raise Exception(f'weight {param} in model {model_id} and {comming_model_id} is not matched')
                    if param_weight.shape == weights[param].shape:
                        if is_init:
                            weights[param].data = similarity * param_weight
                        else:
                            weights[param].data += similarity * param_weight
                    share_boundary = []
                    for i in range(len(weights[param].shape)):
                        share_boundary.append(
                            min(weights[param].shape[i], param_weight.shape[i])
                        )
                    if len(share_boundary) == 0:
                        continue
                    elif len(share_boundary) == 1:
                        if is_init:
                            weights[param].data[:share_boundary[0]] = similarity * param_weight[:share_boundary[0]]
                        else:
                            weights[param].data[:share_boundary[0]] += similarity * param_weight[:share_boundary[0]]
                    elif len(share_boundary) == 2:
                        if is_init:
                            weights[param].data[:share_boundary[0], :share_boundary[1]] = similarity * param_weight[:share_boundary[0], :share_boundary[1]]
                        else:
                            weights[param].data[:share_boundary[0], :share_boundary[1]] += similarity * param_weight[:share_boundary[0], :share_boundary[1]]
                    elif len(share_boundary) == 4:
                        if is_init:
                            weights[param].data[:share_boundary[0], :share_boundary[1], :share_boundary[2], :share_boundary[3]] =\
                                similarity * param_weight[:share_boundary[0], :share_boundary[1], :share_boundary[2], :share_boundary[3]]
                        else:
                            weights[param].data[:share_boundary[0], :share_boundary[1], :share_boundary[2], :share_boundary[3]] +=\
                                similarity * param_weight[:share_boundary[0], :share_boundary[1], :share_boundary[2], :share_boundary[3]]
                    else:
                        raise Exception(f'need to support {len(share_boundary)} dimension of weights')
            return weights, similarity
                    
        else:
            for param in weights.keys():
                param_weight = comming_weights[param]
                if isinstance(param_weight, list):
                    param_weight = np.asarray(param_weight, dtype=np.float32)
                param_weight = torch.from_numpy(param_weight).to(device=device)
                if is_init:
                    weights[param].data = param_weight
                weights[param].data += param_weight
            return weights, 1.0
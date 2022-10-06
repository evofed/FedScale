import logging
from typing import List
import torch
import networkx, onnx
from onnx.helper import printable_graph
from collections import defaultdict
from evofed.lib.net2netlib import *
from copy import deepcopy
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
        
def translate_model(model):
    # 1. translate self.base_model to temporary onnx model
    # 2. parse onnx model to directed acyclic diagram

    name2id = {}
    layername2id = {}

    dummy_input = torch.randn(10, 3, 224, 224)
    torch.onnx.export(model, dummy_input, 'tmp.onnx',
        export_params=True, verbose=0, training=1, do_constant_folding=False)
    onnx_model = onnx.load('tmp.onnx')
    graph = onnx_model.graph
    graph_string = printable_graph(graph)
    start, end = graph_string.find('{'), graph_string.find('}')
    graph_string = graph_string[start+2:end]
    edge_list = graph_string.split('\n')
    input2id = defaultdict(list)
    dag = networkx.DiGraph() #
    
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
        node_id = len(dag.nodes())
        name2id[node.name] = node_id #
        dag.add_node(node_id, attr=node)
        for tensor_input in node.tensor_inputs:
            input2id[tensor_input].append(node_id)
    
    # construct edges
    for node_id in dag.nodes():
        outputs = dag.nodes()[node_id]['attr'].outputs
        for output in outputs:
            for next_id in input2id[output]:
                dag.add_edge(node_id, next_id)
        
    # construct layername2id dict
    for node_id in dag.nodes():
        if dag.nodes()[node_id]['attr'].operator in weight_operator:
            layername2id[dag.nodes()[node_id]['attr'].name] = node_id
    
    return dag, name2id, layername2id

def tensor_crop(weight: torch.Tensor, target_shape: torch.Tensor):
    ts = list(target_shape)
    ts = [':'+str(tss) for tss in ts]
    ts = ",".join(ts)
    return eval(f'deepcopy(weight[{ts}])')

def shape_match(trained_weight: torch.Tensor, weight: torch.Tensor) -> list:
    assert len(trained_weight.shape) == len(weight.shape)
    ts = weight.shape
    trained_is_larger = False
    trained_is_smaller = False
    for dim_t, dim in zip(trained_weight.shape, weight.shape):
        if dim_t > dim:
            trained_is_larger = True
        elif dim_t < dim:
            trained_is_smaller = True
    if trained_is_larger and trained_is_smaller:
        raise Exception(f'error in doing transformation, get weight {trained_weight.shape} and {weight}')
    elif trained_is_larger:
        new_weight = tensor_crop(trained_weight, ts)
    elif trained_is_smaller:
        ratios = []
        for dim_t, dim in zip(trained_weight.shape, weight.shape):
            ratios.append(math.ceil(dim / dim_t))
        new_weight = deepcopy(trained_weight.repeat(ratios))
        new_weight = tensor_crop(new_weight, ts)
    else:
        new_weight = deepcopy(trained_weight)
    return new_weight

class Model_Manager():
    def __init__(self, seed=2333) -> None:
        self.seed = seed
        self.model = []
        self.widen_trajectary = []
        self.deepen_trajectary = []
        self.dags = []
        self.candidate_dags = []
        self.name2id = []
        self.layername2id = []
        self.last_scaled_layer = set()

    def drop_parent_model(self):
        self.model.pop(0)
        self.dags.pop(0)
        self.name2id.pop(0)
        self.layername2id.pop(0)
    
    def load_models(self, models) -> None:
        for i, model in enumerate(models):
            self.model.append(model)
            self.translate_model(i) 
               
    def translate_model(self, model_id: int = -1):
        dag, name2id, layername2id = translate_model(self.model[model_id])
        if model_id == len(self.dags) or model_id + len(self.dags) == -1:
            self.dags.append(dag)
            self.name2id.append(name2id)
            self.layername2id.append(layername2id)
        elif model_id < len(self.dags) and model_id + len(self.dags) >= 0:
            self.dags[model_id] = dag
            self.name2id[model_id] = name2id
            self.layername2id[model_id] = layername2id
        else:
            raise Exception(f"Error when dealing with model {model_id} in model manager")
        
                
    def get_all_nodes(self, model_id):
        if model_id >= len(self.dags):
            self.translate_model(model_id)
        nodes = []
        for node_id in self.dags[model_id].nodes():
            nodes.append(self.dags[model_id].nodes()[node_id]['attr'])
        return nodes

    def get_all_edges(self, model_id):
        if model_id >= len(self.dags):
            self.translate_model(model_id)
        return self.dag.edges()

    def get_convs(self, model_id):
        convs = []
        for node_id in self.dags[model_id].nodes():
            if 'Conv' in self.dags[model_id].nodes()[node_id]['attr'].operator:
                convs.append([node_id, self.dags[model_id].nodes()[node_id]['attr'].name])
        return convs

    def get_weighted_layers(self, model_id):
        layers = []
        for node_id in self.dags[model_id].nodes():
            if self.dags[model_id].nodes()[node_id]['attr'].operator in weight_operator:
                layers.append([node_id, self.dag.nodes()[node_id]['attr'].name])
        return layers

    def get_parents(self, model_id, query_node_id):
        # current only support resnet, mobilenet_v2, alexnet, regnet_x_16gf, vgg19_bn
        # not support shufflenet
        # other nets are not tested yet
        l = []
        for node_id in self.dags[model_id].predecessors(query_node_id):
            node = self.dags[model_id].nodes()[node_id]['attr']
            if node.operator == 'BatchNormalization':
                l.append(node_id)
            if node.operator in weight_operator:
                l.append(node_id)
            else:
                l += self.get_parents(model_id, node_id)
        return l

    def get_children(self, model_id, query_node_id):
        # current only support resnet, mobilenet_v2, alexnet, regnet_x_16gf, vgg19_bn
        # not support shufflenet
        # other nets are not tested yet
        l = []
        for node_id in self.dags[model_id].successors(query_node_id):
            node = self.dags[model_id].nodes()[node_id]['attr']
            if node.operator == 'BatchNormalization':
                l.append(node_id)
            if node.operator in weight_operator:
                l.append(node_id)
            else:
                l += self.get_children(model_id, node_id)
        return l

    def is_conflict(self, model_id, query_node_id):
        conflict_id = -1
        for node_id in self.dags[model_id].successors(query_node_id):
            node = self.dags[model_id].nodes()[node_id]['attr']
            if node.operator in conflict_operator:
                return node_id
            elif node.operator not in weight_operator:
                temp_result = self.is_conflict(model_id, node_id)
                if temp_result != -1:
                    return temp_result
        return conflict_id
    
    def get_add_operand(self, model_id, add_node_id):
        return self.get_parents(model_id, add_node_id)
    
    def get_neighbour(self, model_id, query_node_id):
        # current only support resnet, mobilenet_v2, alexnet, regnet_x_16gf, vgg19_bn
        # not support shufflenet
        # other nets are not tested yet
        l = []
        conflict_node_id = self.is_conflict(model_id, query_node_id)
        while conflict_node_id != -1:
            l += self.get_add_operand(model_id, conflict_node_id)
            conflict_node_id = self.is_conflict(model_id, conflict_node_id)
        return l

    def get_widen_instruction(self, model_id, query_node_id):
        child_convs = set()
        parent_convs = set()
        child_lns = set()
        parent_lns = set()
        bns = set()
        
        children = self.get_children(model_id, query_node_id)
        neighbours = self.get_neighbour(model_id, query_node_id)
        for neighbor in neighbours:
            node = self.dags[model_id].nodes()[neighbor]['attr']
            if node.operator == 'BatchNormalization':
                bns.add(neighbor)
            elif node.operator == 'Gemm':
                parent_lns.add(neighbor)
            else:
                parent_convs.add(neighbor)
            children += self.get_children(model_id, neighbor)
        for child in children:
            node = self.dags[model_id].nodes()[child]['attr']
            if node.operator == 'BatchNormalization':
                bns.add(child)
            elif node.operator == 'Gemm':
                child_lns.add(child)
            else:
                child_convs.add(child)
        return list(child_convs), list(parent_convs), list(bns), list(child_lns), list(parent_lns)
    
    def get_deepen_instruction(self, model_id, query_node_id):
        out_channel, in_channel = None, None

        # get out_channels
        child = self.get_children(model_id, query_node_id)[0]
        child_node = self.dags[model_id].nodes()[child]['attr']
        if child_node.operator == 'BatchNormalization':
            bn = get_model_layer(self.model[model_id], child_node.name)
            out_channel = bn.num_features
        else:
            conv = get_model_layer(self.model[model_id], child_node.name)
            out_channel = conv.in_channels
        
        # get out_channels
        parent_node = self.dags[model_id].nodes()[parent_node]['attr']
        conv = get_model_layer(self.model[model_id], parent_node.name)
        in_channel = conv.out_channels

        return in_channel, out_channel

    def record_trajectary(self, model_id, widened_layers, deepened_layers):
        layers = self.get_weighted_layers()
        layers_onehot = np.array([0 for _ in range(len(layers))])
        def nodeid2convid(node_id):
            for i, conv in enumerate(layers):
                if conv[0] == node_id:
                    return i
            raise Exception(f"not find node {node_id}")
        # record widen trajectary
        if len(self.widen_trajectary) == model_id:
            self.widen_trajectary.append(layers_onehot)
        else:
            self.widen_trajectary[model_id] = layers_onehot
        for node_id in widened_layers:
            conv_id = nodeid2convid(node_id)
            self.widen_trajectary[model_id][conv_id] = 1
        # record deepen trajectary
        if len(self.deepen_trajectary) == model_id:
            self.deepen_trajectary.append(layers_onehot)
        else:
            self.deepen_trajectary[model_id] = layers_onehot
        for node_id in deepened_layers:
            conv_id = nodeid2convid(node_id)
            self.deepen_trajectary[model_id][conv_id] = 1

    def widen_layer(self, model_id, node_id):
        node_name = self.dags[model_id].nodes()[node_id]['attr'].name
        children, parents, bns, ln_children, ln_parents = self.get_widen_instruction(model_id, node_id)
        node = self.dags[model_id].nodes()[node_id]['attr']
        if len(children) == 0 and len(ln_children) == 0:
            print(f'fail to widen {node.name} as it is the last layer')
            return []
        if node.operator == 'Gemm':
            ln_parents.append(node_id)
        elif node_id not in parents:
            parents.append(node_id)
        for child in children:
            node_name = self.dags[model_id].nodes()[child]['attr'].name
            self.model[model_id] = widen_child_conv(
                self.model[model_id], node_name)
        for parent in parents:
            node_name = self.dags[model_id].nodes()[parent]['attr'].name
            self.model[model_id] = widen_parent_conv(
                self.model[model_id], node_name)
        for bn in bns:
            node_name = self.dags[model_id].nodes()[bn]['attr'].name
            self.model[model_id] = widen_bn(
                self.model[model_id], node_name)
        for ln_child in ln_children:
            node_name = self.dags[model_id].nodes()[ln_child]['attr'].name
            self.model[model_id] = widen_child_ln(
                self.model[model_id], node_name
            )
        for ln_parent in ln_parents:
            node_name = self.dags[model_id].nodes()[ln_child]['attr'].name
            node_name = self.dags[model_id].nodes()[ln_parent]['attr'].name
            self.model[model_id] = widen_parent_ln(
                self.model[model_id], node_name
            )
        return parents

    def deepen_layer(self, model_id, node_id):
        node = self.dags[model_id].nodes()[node_id]['attr']
        if node.operator == 'Conv':
            self.model[model_id] = deepen(
                self.model[model_id], node.name
            )
    
    def efficient_model_scale(self, layers: List[str]):
        self.model.append(deepcopy(self.model[-1]))
        widen_layers = []
        deepen_layers = []
        for layer in layers:
            if layer in self.last_scaled_layer:
                deepen_layers.append(layer)
                self.last_scaled_layer.remove(layer)
            else:
                widen_layers.append(layer)
                self.last_scaled_layer.add(layer)
        for layer in widen_layers:
            logging.info(f'widenning layer {layer}')
            node_id = self.layername2id[-1][layer]
            self.widen_layer(-1, node_id)
        for layer in deepen_layers:
            logging.info(f"deepening layer {layer}")
            node_id = self.layername2id[-1][layer]
            self.deepen_layer(-1, node_id)
        self.translate_model(len(self.model)-1)
        
    def get_layers(self, model_id):
        if model_id >= len(self.dags):
            self.translate_model(model_id)
        layers = []
        for node_id in self.dags[model_id].nodes():
            if self.dags[model_id].nodes()[node_id]['attr'].operator in weight_operator:
                layers.append([node_id, self.dags[model_id].nodes()[node_id]['attr'].name])
        return layers
    
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

    def aggregate_weights(self, weights_param, comming_weights_param, model_id, comming_model_id, device, is_init: bool=False):
        """
        softly add weight from comming model to model
        return aggregated weights and aggregation status
        """
        if model_id != comming_model_id:
            similarity = self.get_candidate_similarity(model_id, comming_model_id)
            # check data type
            if weights_param.dtype == torch.int64:
                if is_init:
                    return comming_weights_param
            if len(comming_weights_param.shape) != len(weights_param.shape):
                raise Exception(f'weight {weights_param} in model {model_id} and {comming_model_id} is not matched')
            if weights_param.shape == comming_weights_param.shape:
                if is_init:
                    weights_param = similarity * comming_weights_param
                else:
                    weights_param += similarity * comming_weights_param
            share_boundary = []
            for i in range(len(weights_param.shape)):
                share_boundary.append(
                    min(weights_param.shape[i], comming_weights_param.shape[i])
                )
            if len(share_boundary) == 0:
                raise Exception(f'error aggregate {weights_param} in {is_init}init mode')
            elif len(share_boundary) == 1:
                if is_init:
                    weights_param[:share_boundary[0]] = similarity * comming_weights_param[:share_boundary[0]]
                else:
                    weights_param[:share_boundary[0]] += similarity * comming_weights_param[:share_boundary[0]]
            elif len(share_boundary) == 2:
                if is_init:
                    weights_param[:share_boundary[0], :share_boundary[1]] = similarity * comming_weights_param[:share_boundary[0], :share_boundary[1]]
                else:
                    weights_param[:share_boundary[0], :share_boundary[1]] += similarity * comming_weights_param[:share_boundary[0], :share_boundary[1]]
            elif len(share_boundary) == 4:
                if is_init:
                    weights_param[:share_boundary[0], :share_boundary[1], :share_boundary[2], :share_boundary[3]] =\
                        similarity * comming_weights_param[:share_boundary[0], :share_boundary[1], :share_boundary[2], :share_boundary[3]]
                else:
                    weights_param[:share_boundary[0], :share_boundary[1], :share_boundary[2], :share_boundary[3]] +=\
                        similarity * comming_weights_param[:share_boundary[0], :share_boundary[1], :share_boundary[2], :share_boundary[3]]
            else:
                raise Exception(f'need to support {len(share_boundary)} dimension of weights')
            return weights_param
                    
        else:
            if is_init:
                weights_param = comming_weights_param
            weights_param += comming_weights_param
            return weights_param
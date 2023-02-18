import logging
import math
import os
import sys
from typing import List, Set
import networkx, onnx
from onnx.helper import printable_graph
from collections import defaultdict
from fedscale.core.net2netlib import *
from copy import deepcopy
from thop import profile
import torch
from fedscale.core.logger.aggragation import logDir
import pickle
from dataclasses import dataclass
import numpy as np

omit_operator = ['Identity', 'Constant']
conflict_operator = ['Add', 'Mul']
weight_operator = ['Conv', 'Gemm']
size_sensitive_operator = ['Conv', 'BatchNormalization', 'Gemm']


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
        
def translate_model(model, task):
    # 1. translate self.base_model to temporary onnx model
    # 2. parse onnx model to directed acyclic diagram

    name2id = {}
    layername2id = {}

    if task == "speech":
        dummy_input = torch.randn(10, 1, 32, 32)
    else:
        dummy_input = torch.randn(10, 3, 256, 256)
    torch.onnx.export(model, dummy_input, 'tmp.onnx',
        export_params=True, verbose=0, training=torch.onnx.TrainingMode.TRAINING, do_constant_folding=False)
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

# dummy_input = torch.randn(10, 3, 224, 224)
dataset_input = {
    'femnist': torch.randn(10, 3, 28, 28),
    'openImg': torch.randn(10, 3, 256, 256),
    'google_speech': torch.randn(10, 1, 32, 32),
    'cifar10': torch.randn(10, 3, 32, 32)
}

@dataclass
class ClientRecord:
    training_loss: List[float]
    utility: float

class SuperModel:
    def __init__(self, torch_model, args, rank, last_scaled_layer: Set=None) -> None:
        self.torch_model = torch_model
        self.dag, self.name2id, self.layername2id = \
            translate_model(torch_model, args.task)
        # logging.info(self.layername2id)
        self.macs, self.params = profile(self.torch_model, inputs=(dataset_input[args.data_set],), verbose=False)
        logging.info(f"model {rank} has MACs: {self.macs}")
        if last_scaled_layer is None:
            self.last_scaled_layer = set()
        else:
            self.last_scaled_layer = last_scaled_layer
        self.curr_loss = {}
        self.converged = False
        self.converging = False
        self.model_in_update = 0
        self.gradient_in_update = 0
        self.model_weights = self.torch_model.state_dict()
        self.model_grads_buffer = defaultdict(list)
        self.task_round = 0
        self.train_loss_buffer = []
        self.args = args
        self.rank = rank
        self.last_gradient_weights = []
        self.model_update_size = sys.getsizeof(pickle.dumps(torch_model)) // 1024.0 * 8.0
        self.client_records = {}
        self.count = collections.OrderedDict()
        self.trained_round = 1
        self.inherit = {}
        self.utilities = defaultdict(int)
        self.average_loss = .0
        if rank == 0:
            for layer in self.get_weighted_layers():
                self.inherit[layer[1]] = 0
    
    def load_utilities(self, utilities: dict):
        self.utilities = deepcopy(utilities)

    def load_inherit(self, inherit):
        self.inherit = inherit

    def is_converging(self):
        return self.converging

    def is_converged(self):
        return self.converged

    def reset_curr_loss(self):
        self.curr_loss = {}
    
    def reset_model_in_update(self):
        self.model_in_update = 0
        self.gradient_in_update = 0
        self.count = collections.OrderedDict()

    def assign_task(self, task):
        self.task_round = task

    def assign_one_task(self):
        self.task_round += 1
    
    def reset_task(self):
        self.task_round = 0

    def normal_weight_aggregation(self, results):

        cap = results['cap']
        client_id = results['clientId']
        self.curr_loss[client_id] = results['moving_loss']

        if client_id not in self.client_records:
            self.client_records[client_id] = ClientRecord([], 0)

        self.client_records[client_id].training_loss.append(results['moving_loss'])
        self.model_in_update += 1

        for p in results['update_weight']:
            if self.model_in_update == 1:
                self.model_weights[p].data = results['update_weight'][p]
            else:
                self.model_weights[p].data += results['update_weight'][p]

        self.update_gradient_buffer(cap, results)

        # aggregate model weights
        if self.model_in_update == self.task_round:
            self.trained_round += 1
            for p in self.model_weights:
                d_type = self.model_weights[p].data.dtype
                self.model_weights[p].data = (
                        self.model_weights[p] / float(self.task_round)).to(dtype=d_type)
            for l in self.model_grads_buffer:
                if self.gradient_in_update > 0:
                    # logging.info(f"get {self.gradient_in_update} gradients")
                    self.model_grads_buffer[l][-1] = (
                            self.model_grads_buffer[l][-1] / float(self.gradient_in_update))
            # self.curr_loss = self.curr_loss / self.task_round
            self.train_loss_buffer.append(self.curr_loss)
            self.check_convergence()
            logging.info(f'(DEBUG) gradient buffer of model {self.rank}: {self.model_grads_buffer}')
            logging.info(f'training loss of model {self.rank}: {self.curr_loss}')

    def update_gradient_buffer(self, cap, results):
        if self.gradient_in_update == 0 and cap > self.macs:
            if len(results['grad_dict']) > 0:
                self.gradient_in_update += 1
            for l in results['grad_dict']:
                self.model_grads_buffer[l].append(results['grad_dict'][l])
        elif cap > self.macs:
            if len(results['grad_dict']) > 0:
                self.gradient_in_update += 1
            for l in results['grad_dict']:
                self.model_grads_buffer[l][-1] += results['grad_dict'][l]
                if len(self.model_grads_buffer[l]) > self.args.gradient_buffer_length:
                    self.model_grads_buffer[l].pop(0)

    def soft_weight_aggregation(self, results, model_id, similarity):
        # self.curr_loss += results['moving_loss']
        if self.converging and model_id != self.rank:
            return

        if model_id > self.rank or self.task_round == 0:
            return
        
        # logging.info(f"aggregating model {model_id} into {self.rank} with similarity {similarity}")

        client_id = results['clientId']
        cap = results['cap']

        if client_id not in self.client_records:
            self.client_records[client_id] = ClientRecord([], 0)

        if math.isnan(results['moving_loss']):
            logging.info(f"unexpected training loss {results['moving_loss']}")
            logging.info(f"skip aggregation")
            self.task_round -= 1
            self.weighted_average_weights()
            return


        self.client_records[client_id].training_loss.append(results['moving_loss'])
        self.model_in_update += 1

        # logging.info(f"(DEBUG) tasks required on model {self.rank}: {self.task_round}")
        for p in results['update_weight']:
            # not add hard model parameters
            if self.rank != model_id and ('total_ops' in p or 'total_params' in p):
                continue

            weights = results['update_weight'][p]

            matched = False
            for param_name in self.model_weights:
                param_name_prefix = ".".join(param_name.split('.')[:-1])
                p_prefix = ".".join(p.split('.')[:-1])
                param_name_surfix = param_name.split('.')[-1]
                p_surfix = p.split('.')[-1]
                if p_prefix in param_name_prefix and p_surfix == param_name_surfix:
                    if p_prefix != param_name_prefix:
                        tail = param_name_prefix[:len(p_prefix)]
                        if tail[1] == '0':
                            matched = True
                            # if model_id != self.rank:
                            #     logging.info(f"model {self.rank}: {param_name} <-> model {model_id}: {p}")
                            p = param_name
                            break
                    else:
                        matched = True
                        # if model_id != self.rank:
                        #     logging.info(f"model {self.rank}: {param_name} <-> model {model_id}: {p}")
                        break
                
            if not matched:
                continue

            assert p in self.model_weights

            d_type = self.model_weights[p].data.dtype

            # init count and model weights
            if self.model_in_update == 1 or p not in self.count:
                self.count[p] = torch.zeros_like(self.model_weights[p].data)
                self.model_weights[p].data = torch.zeros_like(self.model_weights[p].data)

            if self.model_weights[p].data.dim() == 0:
                self.count[p] = torch.tensor(0)
                self.model_weights[p].data = weights
            elif self.model_weights[p].data.dim() == 1:
                dim1 = weights.shape[0]
                if self.rank == model_id:
                    self.count[p][:dim1] += torch.ones(dim1)
                    self.model_weights[p].data[:dim1] += weights
                else:
                    self.count[p][:dim1] += torch.ones(dim1) * similarity / float(self.trained_round)
                    self.model_weights[p].data[:dim1] += (weights * similarity / float(self.trained_round)).to(dtype=d_type)
            elif self.model_weights[p].data.dim() == 2:
                dim1, dim2 = weights.shape
                if self.rank == model_id:
                    self.count[p][:dim1, :dim2] += torch.ones((dim1, dim2))
                    self.model_weights[p].data[:dim1, :dim2] += weights
                else:
                    self.count[p][:dim1, :dim2] += torch.ones((dim1, dim2)) * similarity / float(self.trained_round)
                    self.model_weights[p].data[:dim1, :dim2] += (weights * similarity / float(self.trained_round)).to(dtype=d_type)
            elif self.model_weights[p].data.dim() == 4:
                dim1, dim2, dim3, dim4 = weights.shape
                if self.rank == model_id:
                    self.count[p][:dim1, :dim2, :dim3, :dim4] += torch.ones((dim1, dim2, dim3, dim4))
                    self.model_weights[p].data[:dim1, :dim2, :dim3, :dim4] += weights
                else:
                    self.count[p][:dim1, :dim2, :dim3, :dim4] += torch.ones((dim1, dim2, dim3, dim4)) \
                                                                 * similarity / float(self.trained_round)
                    self.model_weights[p].data[:dim1, :dim2, :dim3, :dim4] += (weights * similarity / float(
                        self.trained_round)).to(dtype=d_type)
            else:
                raise Exception(f"does not support dim {self.model_weights[p].data.dim()}")

        # update gradient buffer and current loss if necessary
        if model_id == self.rank:
            self.curr_loss[client_id] = results['moving_loss']
            self.update_gradient_buffer(cap, results)

        # aggregate weights
        self.weighted_average_weights()

    def weighted_average_weights(self):
        if self.model_in_update == self.task_round:
            self.trained_round += 1
            # calculate weighted average weights
            for p in self.model_weights:
                d_type = self.model_weights[p].data.dtype
                if p in self.count and self.count[p].dim() > 0:
                    # prevent divide by zero
                    zero_indexes = (self.count[p] == 0)
                    self.count[p][zero_indexes] = 1.0
                    # logging.info(f"model {self.rank}'s aggregation weights for parameter {p} is\n{self.count[p]}")
                    self.model_weights[p].data = torch.div(
                        self.model_weights[p].data,
                        self.count[p].to(dtype=d_type)).to(dtype=d_type)
            # calculate average gradient
            for l in self.model_grads_buffer:
                if self.gradient_in_update > 0:
                    # logging.info(f"get {self.gradient_in_update} gradients")
                    self.model_grads_buffer[l][-1] = (
                        self.model_grads_buffer[l][-1] / float(self.gradient_in_update)
                    )
            # calculate average loss
            if len(self.curr_loss) > 0:
                avg_loss = .0
                for client_id in self.curr_loss:
                    avg_loss += self.curr_loss[client_id]
                avg_loss /= len(self.curr_loss)
                self.average_loss = avg_loss
                self.train_loss_buffer.append(avg_loss)
                logging.info(f'training loss of model {self.rank}: avg: {avg_loss}: {self.curr_loss}')
                self.check_convergence()
                self.standardize_loss()
            else:
                logging.info(f'training loss of model {self.rank}: no current loss')

    def standardize_loss(self):
        curr_loss_list = np.array([self.curr_loss[client_id] for client_id in self.curr_loss])
        if len(curr_loss_list) == 1:
            for client_id in self.curr_loss:
                self.curr_loss[client_id] = 0
        else:
            for client_id in self.curr_loss:
                self.curr_loss[client_id] = (self.curr_loss[client_id] - np.mean(curr_loss_list)) / np.std(curr_loss_list)
        # DEBUG
        logging.info(f"log standardized loss {self.curr_loss}")
        return np.mean(curr_loss_list)

    def check_convergence(self):
        if len(self.train_loss_buffer) > self.args.window_N + self.args.step_M:
            self.train_loss_buffer.pop(0)
            slope = .0
            for i in range(self.args.window_N):
                slope += abs(self.train_loss_buffer[i] - self.train_loss_buffer[i+self.args.step_M]) / self.args.step_M
            slope /= self.args.window_N
            logging.info(f'current accumulative training loss slope of model {self.rank}: {slope}')
                
            if slope < self.args.transform_threshold:
                self.converging = True
            if self.rank > 10 and slope < self.args.convergent_threshold - 0.0007:
                self.converged = True
            elif self.rank <= 10 and slope < self.args.convergent_threshold:
                self.converged = True

    def terminate(self):
        logging.info(f"terminate the training of model {self.rank}")
        self.save_model()

    def save_last_param(self):
        self.last_gradient_weights = [
            p.data.clone() for p in self.torch_model.parameters()
        ]

    def load_model_weight(self, optimizer):
        self.torch_model.load_state_dict(self.model_weights)
        current_grad_weights = [param.data.clone()
                                for param in self.torch_model.parameters()]
        optimizer.update_round_gradient(
            self.last_gradient_weights, current_grad_weights, self.torch_model)

    def save_model(self):
        model_path = os.path.join(logDir, 'model_'+str(self.rank)+'.pth.tar')
        with open(model_path, 'wb') as model_out:
            pickle.dump(self.torch_model, model_out)

    def get_weighted_layers(self):
        layers = []
        for node_id in self.dag.nodes():
            if self.dag.nodes()[node_id]['attr'].operator in weight_operator:
                layers.append([node_id, self.dag.nodes()[node_id]['attr'].name])
        return layers

    def get_parents(self, query_node_id):
        # current only support resnet, mobilenet_v2, alexnet, regnet_x_16gf, vgg19_bn
        # not support shufflenet
        # other nets are not tested yet
        l = []
        for node_id in self.dag.predecessors(query_node_id):
            node = self.dag.nodes()[node_id]['attr']
            if node.operator == 'BatchNormalization':
                l.append(node_id)
            if node.operator in weight_operator:
                l.append(node_id)
            else:
                l += self.get_parents(node_id)
        return l

    def get_children(self, query_node_id):
        # current only support resnet, mobilenet_v2, alexnet, regnet_x_16gf, vgg19_bn
        # not support shufflenet
        # other nets are not tested yet
        l = []
        for node_id in self.dag.successors(query_node_id):
            node = self.dag.nodes()[node_id]['attr']
            if node.operator == 'BatchNormalization':
                l.append(node_id)
            if node.operator in weight_operator:
                l.append(node_id)
            else:
                l += self.get_children(node_id)
        return l

    def get_add_operand(self, add_node_id):
        return self.get_parents(add_node_id)

    def get_widen_instruction(self, query_node_id):
        child_convs = set()
        parent_convs = set()
        child_lns = set()
        parent_lns = set()
        bns = set()
        
        children = self.get_children(query_node_id)
        neighbours = self.get_neighbour(query_node_id)
        for neighbor in neighbours:
            node = self.dag.nodes()[neighbor]['attr']
            if node.operator == 'BatchNormalization':
                bns.add(neighbor)
            elif node.operator == 'Gemm':
                parent_lns.add(neighbor)
            else:
                parent_convs.add(neighbor)
            children += self.get_children(neighbor)
        for child in children:
            node = self.dag.nodes()[child]['attr']
            if node.operator == 'BatchNormalization':
                bns.add(child)
            elif node.operator == 'Gemm':
                child_lns.add(child)
            else:
                child_convs.add(child)
        return list(child_convs), list(parent_convs), list(bns), list(child_lns), list(parent_lns)
    
    def is_conflict(self, query_node_id):
        conflict_id = -1
        for node_id in self.dag.successors(query_node_id):
            node = self.dag.nodes()[node_id]['attr']
            if node.operator in conflict_operator:
                return node_id
            elif node.operator not in weight_operator:
                temp_result = self.is_conflict(node_id)
                if temp_result != -1:
                    return temp_result
        return conflict_id

    def get_neighbour(self, query_node_id):
        # current only support resnet, mobilenet_v2, alexnet, regnet_x_16gf, vgg19_bn
        # not support shufflenet
        # other nets are not tested yet
        l = []
        conflict_node_id = self.is_conflict(query_node_id)
        while conflict_node_id != -1:
            l += self.get_add_operand(conflict_node_id)
            conflict_node_id = self.is_conflict(conflict_node_id)
        return l

    def widen_layer(self, node_id, new_model):
        node_name = self.dag.nodes()[node_id]['attr'].name
        children, parents, bns, ln_children, ln_parents = self.get_widen_instruction(node_id)
        node = self.dag.nodes()[node_id]['attr']
        if len(children) == 0 and len(ln_children) == 0:
            logging.info(f'fail to widen {node.name} as it is the last layer')
            return new_model
        if node.operator == 'Gemm' and node_id not in ln_parents:
            ln_parents.append(node_id)
        elif node.operator != 'Gemm' and node_id not in parents:
            parents.append(node_id)
        for child in children:
            node_name = self.dag.nodes()[child]['attr'].name
            new_model = widen_child_conv(
                new_model, node_name)
        for parent in parents:
            node_name = self.dag.nodes()[parent]['attr'].name
            new_model = widen_parent_conv(
                new_model, node_name)
        for bn in bns:
            node_name = self.dag.nodes()[bn]['attr'].name
            new_model = widen_bn(
                new_model, node_name)
        for ln_child in ln_children:
            node_name = self.dag.nodes()[ln_child]['attr'].name
            new_model = widen_child_ln(
                new_model, node_name)
        for ln_parent in ln_parents:
            node_name = self.dag.nodes()[ln_parent]['attr'].name
            new_model = widen_parent_ln(
                new_model, node_name)
        return new_model

    def deepen_layer(self, node_id, new_model):
        node = self.dag.nodes()[node_id]['attr']
        if node.operator == 'Conv':
            new_model = deepen(
                new_model, node.name
            )
        if node.operator == 'Gemm':
            new_model = deepen_ln(
                new_model, node.name
            )
        return new_model


    def select_layers_by_gradient(self):
        model_grad_rank = [[l, sum(self.model_grads_buffer[l]) / float(len(self.model_grads_buffer[l]))] for l in self.model_grads_buffer]
        if self.args.selected_ratio == -1: # enforce transformation all
            return [l[0] for l in model_grad_rank]
        model_grad_rank.sort(key=lambda l: l[1])
        max_grad = model_grad_rank[-1][1]
        selected_layers = []
        for l in model_grad_rank:
            if l[1] > self.args.selected_ratio *  max_grad:
                selected_layers.append(l[0])
        return selected_layers

    def model_scale(self, layers: List[str]):
        logging.info(f"selected layers {layers} to scale up at model {self.rank}")
        widen_layers = []
        deepen_layers = []
        scaled_layer = self.last_scaled_layer
        new_model = deepcopy(self.torch_model)
        for layer in layers:
            if layer in self.last_scaled_layer:
                deepen_layers.append(layer)
                scaled_layer.remove(layer)
            else:
                widen_layers.append(layer)
                scaled_layer.add(layer)
        for layer in widen_layers:
            logging.info(f'widenning layer {layer}')
            # print(f'widenning layer {layer}')
            node_id = self.layername2id[layer]
            new_model = self.widen_layer(node_id, new_model)
        for layer in deepen_layers:
            logging.info(f"deepening layer {layer}")
            # print(f'widenning layer {layer}')
            node_id = self.layername2id[layer]
            new_model = self.deepen_layer(node_id, new_model)
        logging.info(new_model)
        # self.dag, self.name2id, self.layername2id = \
        #     translate_model(self.torch_model)
        return new_model, scaled_layer

    def model_deepen(self, layers: List[str]):
        logging.info(f"selected layers {layers} to scale up at model {self.rank}")
        new_model = deepcopy(self.torch_model)
        for layer in layers:
            logging.info(f"deepening layer {layer}")
            # print(f'widenning layer {layer}')
            node_id = self.layername2id[layer]
            new_model = self.deepen_layer(node_id, new_model)
        logging.info(new_model)
        # self.dag, self.name2id, self.layername2id = \
        #     translate_model(self.torch_model)
        return new_model, layers

    def get_size_sensitive_layers(self):
        layers = []
        for node_id in self.dag.nodes():
            if self.dag.nodes()[node_id]['attr'].operator in size_sensitive_operator:
                layers.append([node_id, self.dag.nodes()[node_id]['attr'].name])
        return layers

    def model_width_scale(self, ratio: float=0.5):
        # modify all weighted layers, including conv, bn, and ln
        layers = self.get_size_sensitive_layers()
        new_model = deepcopy(self.torch_model)
        for idx, layer in enumerate(layers):
            node_id, layer_name = layer
            node = self.dag.nodes()[node_id]['attr']
            if node.operator == 'Gemm':
                new_model = scale_ln(new_model, layer_name, ratio, idx==len(layers)-1)
            elif node.operator == 'Conv':
                new_model = scale_conv(new_model, layer_name, ratio, idx==0, idx==len(layers)-1)
            elif node.operator == 'BatchNormalization':
                new_model = scale_bn(new_model, layer_name, ratio)
        return new_model


class Model_Manager():
    def __init__(self, init_model, args) -> None:
        self.models = []
        self.args = args
        self.add_model(init_model)
        self.similarities = [[1]]

    def add_model(self, torch_model):
        self.models.append(SuperModel(torch_model, self.args, len(self.models), set()))

    def get_latest_model(self):
        return self.models[-1].torch_model
    
    def get_all_models(self):
        models = []
        for model in self.models:
            models.append(model.torch_model)
    
    def model_scale_single(self):
        layers = self.models[-1].select_layers_by_gradient()
        new_model, last_scaled_layer = self.models[-1].model_scale(layers)
        # drop the last model
        # TODO: do not drop the last model
        self.models[-1] = None
        self.models.append(SuperModel(new_model, self.args, len(self.models), last_scaled_layer))
        return self.models[-1].torch_model
    
    def model_scale(self):
        super_model = self.models[-1]

        assert isinstance(super_model, SuperModel)

        layers = super_model.select_layers_by_gradient()
        new_model, last_scaled_layer = super_model.model_scale(layers)

        new_super_model = SuperModel(new_model, self.args, len(self.models), last_scaled_layer)
        new_inherit = self.generate_inherit(new_super_model, super_model)

        new_super_model.load_inherit(new_inherit)
        new_super_model.load_utilities(super_model.utilities)

        self.models.append(new_super_model)

        self.similarities.append([])

        # update similarity
        self.update_similarities()

    def random_scale(self):
        super_model = self.models[-1]

        assert isinstance(super_model, SuperModel)

        import random
        layers = random.sample(super_model.get_weighted_layers(), k=2)
        layers = [layer[1] for layer in layers]
        new_model, last_scaled_layer = super_model.model_scale(layers)

        new_super_model = SuperModel(new_model, self.args, len(self.models), last_scaled_layer)
        new_inherit = self.generate_inherit(new_super_model, super_model)
        print(f"model{len(self.models)}: {new_inherit}")
        new_super_model.load_inherit(new_inherit)

        self.models.append(new_super_model)

        self.similarities.append([])

        self.update_similarities()

    def update_similarities(self):
        self.similarities = []
        for i in range(len(self.models)):
            self.similarities.append([])
            for j in range(len(self.models)):
                self.similarities[-1].append(self.get_similarity(i, j))

        for i in range(len(self.similarities)):
            max_similarity = max(self.similarities[i])
            for j in range(len(self.similarities[i])):
                self.similarities[i][j] /= max_similarity

    def get_similarity(self, i: int, j: int):
        larger = max([i, j])
        smaller = min([i, j])

        inherit = self.models[larger].inherit

        similarity = 0
        for layer_name in inherit:
            rank = inherit[layer_name]
            if rank <= smaller:
                score = self.get_layer_score(layer_name, self.models[smaller], self.models[larger])
                similarity += score
            else:
                similarity += -1

        return max([0, similarity])

    def get_layer_score(self, layer_name: str, small_model: SuperModel, large_model: SuperModel):
        small_layers = small_model.get_weighted_layers()
        large_weight = get_model_layer_weight(large_model.torch_model, layer_name)
        small_weight = None
        for layer in small_layers:
            if layer[1] in layer_name:
                small_weight = get_model_layer_weight(small_model.torch_model, layer[1])
        assert small_weight is not None
        assert len(large_weight.shape) == len(small_weight.shape)
        return self.get_params(small_weight) / self.get_params(large_weight)

    def get_params(self, large_weight):
        num_params = 1
        for dim in large_weight.shape:
            num_params *= dim
        return num_params

    def generate_inherit(self, new_super_model, super_model):
        layers = super_model.get_weighted_layers()
        layer_names = [layer[1] for layer in layers]
        new_layers = new_super_model.get_weighted_layers()
        new_layer_names = [layer[1] for layer in new_layers]
        inherit = super_model.inherit
        new_inherit = {}
        missed_layers = []
        for new_layer in new_layers:
            if new_layer[1] in layer_names:
                new_inherit[new_layer[1]] = inherit[new_layer[1]]
            else:
                missed_layers.append(new_layer)
        # print(missed_layers)
        assert len(missed_layers) % 2 == 0
        for missed_layer in missed_layers:
            if missed_layer[1][-1] != '0':
                new_inherit[missed_layer[1]] = len(self.models)
                continue
            missed_layer_name = missed_layer[1]
            inherit_rank = -1
            for layer_name in layer_names:
                if layer_name + '.0' == missed_layer_name:
                    inherit_rank = inherit[layer_name]
            new_inherit[missed_layer_name] = inherit_rank
            if inherit_rank == -1:
                raise Exception
        return new_inherit

    def model_width_scale(self, ratio: float=0.5, inplace: bool=False):
        new_model = self.models[-1].model_width_scale(ratio)
        if inplace:
            self.models[-1] = SuperModel(new_model, self.args, 0, set())
        else:
            self.models.append(SuperModel(new_model, self.args, len(self.models), set()))

    def get_candidate_layers(self, model_id):
        assert self.models[model_id] != None
        return self.models[model_id].get_weighted_layers()

    def get_model_weights(self):
        model_weights = []
        for super_model in self.models:
            if super_model:
                model_weights.append(super_model.torch_model.state_dict())
        return model_weights
        

    def reset_all_curr_loss(self):
        for super_model in self.models:
            if super_model:
                super_model.reset_curr_loss()

    def weight_aggregation(self, results, model_id):
        if not self.args.soft_agg:
            self.models[model_id].normal_weight_aggregation(results)
            if self.models[model_id].converged:
                self.models[model_id].terminate()
                self.models[model_id] = None
        else:
            for idx, model in enumerate(self.models):
                assert isinstance(model, SuperModel)
                if model.converged:
                    model.terminate()
                    continue
                similarity = self.similarities[model_id][idx]
                model.soft_weight_aggregation(results, model_id, similarity)
                    # self.models[idx] = None


    def save_last_param(self):
        for super_model in self.models:
            if super_model:
                super_model.save_last_param()

    def load_model_weight(self, optimizer):
        for super_model in self.models:
            if super_model:
                super_model.load_model_weight(optimizer)

    def save_models(self):
        for super_model in self.models:
            if super_model:
                super_model.save_model()

    def is_converging(self):
        return self.models[-1].is_converging()

    def reset_model_in_update(self):
        for super_model in self.models:
            if super_model:
                super_model.reset_model_in_update()

    def assign_tasks_naive(self, clients_to_run):
        assignment = {}
        model_training = []
        for Id, super_model in enumerate(self.models):
            if super_model:
                super_model.assign_task(len(clients_to_run))
                for client in clients_to_run:
                    assignment[client] = Id
                model_training.append(Id)
        return assignment, model_training

    def get_all_macs(self):
        macs = []
        for super_model in self.models:
            if super_model is not None:
                macs.append(super_model.macs)
        return macs

    def reset_tasks(self):
        for super_model in self.models:
            if super_model:
                super_model.reset_task()
    
    def assign_tasks_hardware(self, clients_to_run, clients_cap):
        assignment = {}
        model_training = set()
        self.reset_tasks()
        for client in clients_to_run:
            for Id, super_model in enumerate(reversed(self.models)):
                if super_model.macs <= clients_cap[client]:
                    super_model.assign_one_task()
                    assignment[client] = len(self.models) - Id - 1
                    model_training.add(len(self.models) - Id - 1)
                    break
            if client not in assignment:
                super_model.assign_one_task()
                assignment[client] = 0
                model_training.add(0)
        if self.args.soft_agg:
            self.models[-1].assign_task(len(clients_to_run))
        # (DEBUG) check model tasks
        tasks = {}
        for idx, model in enumerate(self.models):
            tasks[idx] = model.task_round
        logging.info(f"tasks: {tasks}")
        logging.info(f"MACs of outstanding models {self.get_all_macs()}")
        # logging.info(f"MACs of selected clients {clients_cap}")
        return assignment, list(model_training)

    def assign_tasks_hybrid(self, clients_to_run, clients_cap):
        assignment = {}
        model_training = set()
        self.reset_tasks()
        for client_id in clients_to_run:
            candidate_models = []
            for model_id, super_model in enumerate(self.models):
                assert isinstance(super_model, SuperModel)
                if super_model.macs <= clients_cap[client_id]:
                    candidate_models.append([model_id, super_model])
            if len(candidate_models) == 0:
                assignment[client_id] = 0
                model_training.add(0)
                continue
            else:
                # check if the candidate models are converged:
                not_converged_candidate_models = []
                for model_id, super_model in candidate_models:
                    assert isinstance(super_model, SuperModel)
                    if not super_model.is_converged():
                        not_converged_candidate_models.append([model_id, super_model])
                if len(not_converged_candidate_models) > 0:
                    # not all candidate models are converged
                    # select model from those non-converged models
                    models_ids, probabilities = self.get_probabilities(client_id, not_converged_candidate_models)
                    decided = False
                    while not decided:
                        decision = np.random.multinomial(1, probabilities)
                        assert sum(decision) == 1
                        for model_id, d in zip(models_ids, decision):
                            if d == 1:
                                super_model = self.models[model_id]
                                assert isinstance(super_model, SuperModel)
                                if super_model.converged:
                                    break
                                else:
                                    assignment[client_id] = model_id
                                    model_training.add(model_id)
                                    decided = True
                                    break
                else:
                    # all candidate models are converged
                    # train the selected for other more complex models
                    models_ids, probabilities = self.get_probabilities(client_id, candidate_models)
                    decision = np.random.multinomial(1, probabilities)
                    assert sum(decision) == 1
                    for model_id, d in zip(models_ids, decision):
                        if d == 1:
                            super_model = self.models[model_id]
                            assert isinstance(super_model, SuperModel)
                            assignment[client_id] = model_id
                            break

        for client_id in assignment:
            model_id = assignment[client_id]
            for super_model in self.models[model_id:]:
                assert isinstance(super_model, SuperModel)
                if super_model.rank in model_training:
                    if super_model.is_converging():
                        if super_model.rank == model_id:
                            super_model.assign_one_task()
                    else:
                        super_model.assign_one_task()
        
        # DEBUG
        for super_model in self.models:
            assert isinstance(super_model, SuperModel)
            logging.info(f"model {super_model.rank} has {super_model.task_round} tasks")      
        return assignment, model_training

    def get_probabilities(self, client_id, candidate_models):
        utilities = {}
        for model_id, super_model in candidate_models:
            utilities[model_id] = super_model.utilities[client_id]
            if utilities[model_id] == 0:
                utilities[model_id] = 0.1
                # normalize utility
        base = .0
        max_shift = max([utilities[model_id] for model_id in utilities])
        for model_id in utilities:
            utilities[model_id] -= max_shift
        for model_id in utilities:
            base += math.exp(utilities[model_id])
        for model_id in utilities:
            utilities[model_id] = math.exp(utilities[model_id]) / base
        # logging.info(f"normalized utility for client {client_id}: {utilities}")
                # generate probabilities
        utilities_list = [[model_id, utilities[model_id]] for model_id in utilities]
        models_ids = []
        probabilities = []
        for model_id, probability in utilities_list:
            models_ids.append(model_id)
            probabilities.append(probability)
        # logging.info(f"soft clustering probabilities {probabilities}")
        return models_ids, probabilities

    def update_utility(self, assignment, clients_cap):
        logging.info(f"Updating clients' utilities on each model")
        for client_id in assignment:
            model_id = assignment[client_id]
            super_model = self.models[model_id]
            assert isinstance(super_model, SuperModel)
            if super_model.is_converged():
                logging.info(f"skip update utility of client {client_id} at model {model_id} as model {model_id} is converged")
                continue
            loss = self.models[model_id].curr_loss[client_id]
            candidate_models = []
            client_cap = clients_cap[client_id]
            for idx, super_model in enumerate(self.models):
                assert isinstance(super_model, SuperModel)
                if super_model.macs <= client_cap:
                    candidate_models.append([idx, super_model])
            if len(candidate_models) == 0:
                # logging.info(f"no candidate model for client {client_id}, using the default model (0)")
                candidate_models = [[model_id, self.models[model_id]]]
                continue
            # logging.info(f"calculating utility for client on {len(candidate_models)} models")
            for candidate_model in candidate_models:
                reward = loss * self.similarities[model_id][candidate_model[0]]
                # logging.info(f"Reward of client {client_id} model {candidate_model[1].rank} {reward}")
                candidate_model[1].utilities[client_id] -= reward
                # logging.info(f"Utility of client {client_id} model {candidate_model[1].rank} {candidate_model[1].utilities[client_id]}")

    def get_all_models(self):
        models = []
        for super_model in self.models:
            if super_model:
                models.append(super_model.torch_model)
            else:
                models.append(None)
        return models

    def get_active_model_ids(self):
        active_model_ids = []
        for i, super_model in enumerate(self.models):
            assert isinstance(super_model, SuperModel)
            if not super_model.is_converged():
                active_model_ids.append(i)
        return active_model_ids
    
    def get_model_update_size(self, Id):
        if self.models[Id]:
            return self.models[Id].model_update_size
        else:
            return self.models[-1].model_update_size
    
    def get_model_update_size_all(self):
        size = .0
        for super_model in self.models:
            if super_model:
                size += super_model.model_update_size
        return size

    def check_status(self):
        # checking the status of each super models
        # is_converging, converged, trained rounds, gradient buffer length, utilities, similarity
        # logging.info(f"(MODEL MANAGER STATUS) #models: {len(self.models)}")
        # check is_converging:
        is_converging_models = []
        for super_model in self.models:
            assert isinstance(super_model, SuperModel)
            if super_model.is_converging():
                is_converging_models.append(super_model.rank)
        # logging.info(f"(MODEL MANAGER STATUS) #converging models {len(is_converging_models)}: {is_converging_models}")
        # check is_converged:
        is_converged_models = []
        for super_model in self.models:
            assert isinstance(super_model, SuperModel)
            if super_model.is_converged():
                is_converged_models.append(super_model.rank)
        # logging.info(f"(MODEL MANAGER STATUS) #converged models {len(is_converged_models)}: {is_converged_models}")
        # check similarities:
        # for idx, model_similarity in enumerate(self.similarities):
            # logging.info(f"(MODEL MANAGER STATUS) similarity of model {idx}: {model_similarity}")
        # trained rounds:
        trained_rounds = []
        for super_model in self.models:
            assert isinstance(super_model, SuperModel)
            # logging.info(f"(MODEL MANAGER STATUS) trained rounds of model {super_model.rank}: {super_model.trained_round}")
            trained_rounds.append(super_model.trained_round)
        # gradient_buffer_length:
        # for super_model in self.models:
        #     assert isinstance(super_model, SuperModel)
        #     logging.info(f"(MODEL MANAGER STATUS) length of gradient buffer of model {super_model.rank}: {len(super_model.model_grads_buffer)}")
        # utilities
        utilities = []
        for super_model in self.models:
            assert isinstance(super_model, SuperModel)
            # logging.info(f"(MODEL MANAGER STATUS) utilities of model {super_model.rank}: {super_model.utilities}")
            utilities.append(super_model.utilities)
        # curr_loss:
        curr_loss = []
        avg_loss = []
        for super_model in self.models:
            assert isinstance(super_model, SuperModel)
            # logging.info(f"(MODEL MANAGER STATUS) normalized loss of model {super_model.rank}: {super_model.curr_loss}")
            curr_loss.append(super_model.curr_loss)
            avg_loss.append(super_model.average_loss)

        return len(self.models), len(is_converging_models), len(is_converged_models), trained_rounds, utilities, curr_loss, avg_loss

    def aggregate_weights(self, weights_param, comming_weights_param, model_id, comming_model_id, device, is_init: bool=False):
        """
        softly add weight from comming model to model
        return aggregated weights and aggregation status
        """
        if model_id != comming_model_id:
            # similarity = self.get_candidate_similarity(model_id, comming_model_id)
            similarity = 1 # disable soft aggregation
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
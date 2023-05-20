import pickle
import torchvision.models as models
from fedscale.utils.models.specialized.resnet_speech import resnet34
from fedscale.utils.models.evofed.small_resnet18_speech import small_resnet18_speech
import os
from thop import profile
import numpy as np
import fedscale.utils.models.evofed.nasbench as nasbench
import torch, csv



config = {'name': 'infer.tiny', 'N':0, 'C':1, 'arch_str': '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|', 'num_classes': 10}

outputClass = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47, 'amazon': 5,
               'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5, 'inaturalist': 1010
               }

dataset_info = {
    'femnist': (1, 3, 28, 28),
    'cifar10': (1, 3, 32, 32),
    'speech': (1, 1, 32, 32)
}

model_zoo = {
    'resnet_18': models.resnet18(),
    'resnet_34_speech': resnet34(num_classes=outputClass['google_speech'], in_channels=1),
    'resnet_s_speech': small_resnet18_speech(),
    'mobilenetv3_s': models.mobilenet_v3_small(),
    'nasbench': nasbench.get_cell_based_tiny_net(config)
}

small_model = {
    'femnist': 'nasbench',
    'cifar10': 'mobilenetv3_s',
    'speech': 'resnet_s_speech'
}

large_model = {
    'femnist': 'resnet_18',
    'cifar10': 'resnet_18',
    'speech': 'resnet_34_speech'
}

computation = {}
communication  = {}
with open('../dataset/data/device_info/client_device_capacity', 'rb') as f:
    device_cap = pickle.load(f)

# normalize computation
comp_caps = []
for client_id in device_cap:
    comp_cap = float(device_cap[client_id]['computation'])
    comp_caps.append(comp_cap)
    computation[client_id] = comp_cap
min_cap = min(comp_caps)
max_cap = max(comp_caps)
for client_id in computation:
    computation[client_id] = (computation[client_id] - min_cap) / (max_cap - min_cap)

# normalize communication
comm_caps = []
for client_id in device_cap:
    comm_cap = float(device_cap[client_id]['communication'])
    comm_caps.append(comm_cap)
    communication[client_id] = comm_cap
min_cap = min(comm_caps)
max_cap = max(comm_caps)
for client_id in communication:
    communication[client_id] = (communication[client_id] - min_cap) / (max_cap - min_cap)

capacities = {}
caps = []
for client_id in computation:
    capacities[client_id] = communication[client_id] + computation[client_id]
    caps.append(capacities[client_id])

# normalize capacities
max_cap = max(caps)
min_cap = min(caps)
for client_id in capacities:
    capacities[client_id] = (capacities[client_id] - min_cap) / (max_cap - min_cap)

for dataset in dataset_info:
    print(dataset)
    if not os.path.exists(dataset+"_cap"):
        os.mkdir(dataset+"_cap")
    small_model_str = small_model[dataset]
    large_model_str = large_model[dataset]
    dummy_input = torch.randn(dataset_info[dataset])
    min_macs, min_params = profile(model_zoo[small_model_str], inputs=(dummy_input, ), verbose=False)
    print(min_macs, min_params)
    max_macs, max_params = profile(model_zoo[large_model_str], inputs=(dummy_input, ), verbose=False)
    print(max_macs, max_params)
    client_params = []
    client_ids = []
    client_macs = []
    for client_id in computation:
        client_params.append((max_params - min_params) * capacities[client_id] + min_params)
        client_macs.append((max_macs - min_macs) * capacities[client_id] + min_macs)
        client_ids.append(client_id)
    records = list(zip(client_ids, client_macs, client_params))
    with open(f'{dataset}_cap/realistic.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'macs', 'params'])
        writer.writerows(records)

       
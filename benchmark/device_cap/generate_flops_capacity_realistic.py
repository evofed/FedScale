import pickle
import torchvision.models as models
from fedscale.utils.models.specialized.resnet_speech import resnet34
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
    'femnist': (1, 3, 28, 28)
    # 'openimage': (1, 3, 256, 256),
    # 'speech': (32, 32)
}

model_zoo = {
    'resnet18': models.resnet18(),
    # 'resnet34': resnet34(num_classes=outputClass['google_speech'], in_channels=1),
    'shufflenet': models.shufflenet_v2_x2_0(),
    # 'mobilenet': models.mobilenet_v2(),
    'nasbench': nasbench.get_cell_based_tiny_net(config)
}

computation = {}
with open('../dataset/data/device_info/client_device_capacity', 'rb') as f:
    device_cap = pickle.load(f)

caps = []
for client_id in device_cap:
    cap = float(device_cap[client_id]['computation'])
    caps.append(cap)
    computation[client_id] = cap
min_cap = min(caps)
max_cap = max(caps)
for client_id in computation:
    computation[client_id] = (computation[client_id] - min_cap) / (max_cap - min_cap)

for dataset in dataset_info:
    if not os.path.exists(dataset):
        os.mkdir(dataset)
    dummy_input = torch.randn(dataset_info[dataset])
    min_macs, min_params = profile(model_zoo['nasbench'], inputs=(dummy_input, ), verbose=False)
    print(f"min MACs: {min_macs}, min params: {min_params}")
    for model_name in ['resnet18', 'shufflenet']:
        max_macs, max_params = profile(model_zoo['resnet18'], inputs=(dummy_input, ), verbose=False)
        print(f"max MACs: {max_macs}, max MACs: {max_params}")
        client_params = []
        client_ids = []
        client_macs = []
        for client_id in computation:
            client_params.append((max_params - min_params) * computation[client_id] + min_params)
            client_macs.append((max_macs - min_macs) * computation[client_id] + min_macs)
            client_ids.append(client_id)
        records = list(zip(client_ids, client_macs, client_params))
        with open(f'{dataset}/{model_name}-realistic.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'macs', 'params'])
            writer.writerows(records)

       
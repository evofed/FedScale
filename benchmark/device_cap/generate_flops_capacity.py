import torchvision.models as models
from fedscale.utils.models.specialized.resnet_speech import resnet34
from fedscale.utils.models.evofed.small_resnet18_speech import small_resnet18_speech
from fedscale.utils.models.evofed.small_resnet18 import small_resnet18
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
    'cifar': 'mobilenetv3_s',
    'speech': 'resnet_s_speech'
}

large_model = {
    'femnist': 'resnet_18',
    'cifar': 'resnet_18',
    'speech': 'resnet_34_speech'
}

n_client = 500000

for dataset in dataset_info:
    if not os.path.exists(dataset+"_cap"):
        os.mkdir(dataset+"_cap")
    small_model_str = small_model[dataset]
    large_model_str = large_model[dataset]
    dummy_input = torch.randn(dataset_info[dataset])
    min_macs, min_params = profile(model_zoo[small_model_str], inputs=(dummy_input, ), verbose=False)
    print(min_macs, min_params)
    max_macs, max_params = profile(model_zoo[large_model_str], inputs=(dummy_input, ), verbose=False)
    print(max_macs, max_params)
    # normal distribution
    client_cap_macs = np.random.normal(loc=(min_macs+max_macs)/2, scale=(min_macs+max_macs)/2/2, size=n_client)
    client_cap_params = np.random.normal(loc=(min_params+max_params)/2, scale=(min_params+max_params)/2/2, size=n_client)
    client_ids = list(range(0,n_client))
    records = list(zip(client_ids, client_cap_macs, client_cap_params))
    with open(f'{dataset}_cap/normal.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'macs', 'params'])
        writer.writerows(records)
    
    # uniform distribution
    client_cap_macs = np.random.uniform(low=min_macs, high=max_macs, size=n_client)
    client_cap_params = np.random.uniform(low=min_macs, high=max_macs, size=n_client)
    client_ids = list(range(0,n_client))
    records = list(zip(client_ids, client_cap_macs, client_cap_params))
    with open(f'{dataset}_cap/uniform.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'macs', 'params'])
        writer.writerows(records)
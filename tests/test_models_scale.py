import fedscale.core.nasbench as nasbench
from fedscale.core.model_manager import Model_Manager
import torch
import torchvision.models as models

config = {'name': 'infer.tiny', 'N':0, 'C':1, 'arch_str': '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|', 'num_classes': 10}
model = nasbench.get_cell_based_tiny_net(config)
manager = Model_Manager(model, candidate_capacity=5)
manager.translate_base_model()
manager.base_model_scale()
dummy_input = torch.randn((10, 3, 244, 244))
print(model(dummy_input))
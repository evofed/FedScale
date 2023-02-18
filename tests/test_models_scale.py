from torchvision import models
from fedscale.core.model_manager import SuperModel
import torch
from torchinfo import summary

from argparse import Namespace
ns = Namespace(**{"task": "vision", "data_set": "cifar10"})

def dry_forward(torch_model, fake_input):
    try:
        torch_model(fake_input)
        print('successful')
    except Exception as e:
        print(f'error in forward the model as {e}')
        print(torch_model)


model_zoo = {
    # 'resnet18': models.resnet18()
    'mobilenet': models.mobilenet_v3_small(),
    # 'alexnet': models.alexnet(),
    # 'regnet': models.regnet_x_16gf(),
    # 'vgg': models.vgg19_bn(),
    # 'resnet152': models.resnet152()
}

dummy_input = torch.randn((10, 3, 28, 28))
for model in model_zoo.keys():
    print(f'testing {model}')
    # print(model_zoo[model])
    super_model = SuperModel(model_zoo[model], rank=0, args=ns)
    summary(super_model.torch_model, input_size=((10, 3, 28, 28)))
    layers = super_model.get_weighted_layers()
    layers = [l[1] for l in layers]
    print(f"widening {layers}")
    new_model, _ = super_model.model_scale(layers=layers)
    summary(new_model, input_size=((10, 3, 28, 28)))
    print(f"deepening {layers}")
    new_model, _ = super_model.model_deepen(layers=layers)
    summary(new_model, input_size=((10, 3, 28, 28)))
    print('-'*40)
    print('='*40)
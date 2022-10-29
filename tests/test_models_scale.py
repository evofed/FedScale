from torchvision import models
from fedscale.core.model_manager import SuperModel
import torch

def dry_forward(torch_model, fake_input):
    try:
        torch_model(fake_input)
        print('successful')
    except Exception as e:
        print(f'error in forward the model as {e}')
        # print(torch_model)


model_zoo = {
    'resnet18': models.resnet18()
    # 'mobilenet': models.mobilenet_v2(),
    # 'alexnet': models.alexnet(),
    # 'regnet': models.regnet_x_16gf(),
    # 'vgg': models.vgg19_bn(),
    # 'resnet152': models.resnet152()
}

for model in model_zoo.keys():
    print(f'testing {model}')
    # print(model_zoo[model])
    super_model = SuperModel(model_zoo[model])
    layers = super_model.get_weighted_layers()
    layers = [l[1] for l in layers][2:4]
    print(f"widening {layers}")
    super_model.model_scale(layers=layers)
    print(super_model.torch_model)
    print(f"deepening {layers}")
    super_model.model_scale(layers=layers)
    print(super_model.torch_model)
    dummy_input = torch.randn((10, 3, 244, 244))
    dry_forward(super_model.torch_model, dummy_input)
    print('-'*40)
    print('='*40)
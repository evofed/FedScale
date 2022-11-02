import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch

class Naive_CNN(nn.Module):
    """
    a naive convolutional 3-layer neural network for cifar10 & openimage
    Conv2D(32) -> maxpool(3,3) -> Conv2D(64) -> maxpool(3,3) -> Conv2D(10) -> GlobalAveragePooling2D (GAP) -> FC(10)
    """
    def __init__(self,
                 channels,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(Naive_CNN, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.features = nn.Sequential()
        for i, out_channels in enumerate(channels):
            stage = nn.Sequential()
            stage.add_module(
                f"cnn{i}",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    groups=1,
                    bias=True
                )
            )
            in_channels = out_channels
            stage.add_module(
                f"pool{i}",
                nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=0
                )
            )
            self.features.add_module(f"stage{i}", stage)
        self.output = nn.Linear(
            in_features=in_channels * 3 * 3,
            out_features=num_classes
        )

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

def get_ncnn(dataset='cifar10', **kwargs):
    if dataset == 'cifar10':
        channels = [32, 64, 10]
        net = Naive_CNN(
            channels=channels,
            **kwargs
        )
    else:
        raise NotImplementedError

    return net

def ncnn_cifar(**kwargs):
    return get_ncnn('cifar10', **kwargs)

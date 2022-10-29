from train_cifar import Cifar_Runner_CPU
import fedscale.utils.models.evofed.nasbench as nasbench
import torch
torch.set_num_threads(35)
config = {'name': 'infer.tiny', 'N':0, 'C':1, 'arch_str': '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|', 'num_classes': 10}
model = nasbench.get_cell_based_tiny_net(config)
runner = Cifar_Runner_CPU()
runner.train(model)

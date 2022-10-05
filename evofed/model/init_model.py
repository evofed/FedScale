import fedscale.core.fllibs as fllibs
from .nasbench import get_cell_based_tiny_net

def init_model(task, model, dataset):
    if model == 'nasbench201':
        config = {
            'name': 'infer.tiny', 
            'N':0, 
            'C':1, 
            'arch_str': '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|', 
            'num_classes': fllibs.outputClass[dataset]
        }
        model = get_cell_based_tiny_net(config)
        return model
    else:
        return fllibs.init_model()
import torch.nn as nn
import torch, sys, pickle

def weight_reset(m):
    print(type(m))
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()

def reset_model(model: torch.nn.Module):
    model.apply(weight_reset)
    return model

if __name__ == '__main__':
    model_id = sys.argv[1]
    with open(f"../../checkpoints/model_{model_id}_t.pth.tar", 'rb') as f:
        model = pickle.load(f)
    model = reset_model(model)
    with open(f"../../checkpoints/model_{model_id}_r.pth.tar", "wb") as f:
        pickle.dump(model, f)
import numpy as np
import torch

def split_cifar10(dataset, num_users):
    classes_size = dataset.classes_size
    dataset = dataset
    label = np.array(dataset.target)
    non_iid_n = 2
    shard_per_user = non_iid_n
    data_split = {i: [] for i in range(num_users)}
    label_idx_split = {}
    label_split = None
    for i in range(len(label)):
        label_i = label[i].item()
        if label_i not in label_idx_split:
            label_idx_split[label_i] = []
        label_idx_split[label_i].append(i)
    shard_per_class = int(shard_per_user * num_users / classes_size)

    for label_i in label_idx_split:
        label_idx = label_idx_split[label_i]
        num_leftover = len(label_idx) % shard_per_class
        leftover = label_idx[-num_leftover:] if num_leftover > 0 else []
        new_label_idx = np.array(label_idx[:-num_leftover]) if num_leftover > 0 else np.array(label_idx)
        new_label_idx = new_label_idx.reshape((shard_per_class, -1)).tolist()
        for i, leftover_label_idx in enumerate(leftover):
            new_label_idx[i] = np.concatenate([new_label_idx[i], [leftover_label_idx]])
        label_idx_split[label_i] = new_label_idx

    if label_split is None:
        label_split = list(range(classes_size)) * shard_per_class
        label_split = torch.tensor(label_split)[torch.randperm(len(label_split))].tolist()
        label_split = np.array(label_split).reshape((num_users, -1)).tolist()
        for i in range(len(label_split)):
            label_split[i] = np.unique(label_split[i]).tolist()

    for i in range(num_users):
        for label_i in label_split[i]:
            idx = torch.arange(len(label_idx_split[label_i]))[torch.randperm(len(label_idx_split[label_i]))[0]].item()
            data_split[i].extend(label_idx_split[label_i].pop(idx))

    return data_split
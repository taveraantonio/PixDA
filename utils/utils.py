import os
import torch


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_onehot_vector(args, labels):
    labels = labels.long().unsqueeze(1)
    labels[labels == 255] = args.num_classes
    bs, _, h, w = labels.size()
    nc = args.num_classes + 1
    input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, labels, 1.0)[0][:args.num_classes].unsqueeze(0)
    return input_semantics

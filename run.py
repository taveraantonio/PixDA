import os
import os.path as osp
import numpy as np
import torch
from torch.utils import data
import random
import torch.backends.cudnn as cudnn

from models.deeplabV2 import deeplabv2_resnet
from models.discriminator import PixDADiscriminator, FCDiscriminator
from models.utils import change_normalization_layer
from scripts.train import train_pixda
from scripts.eval import test
from utils.config import Options
from utils.metrics import StreamSegMetrics
from dataset.utils import find_dataset_using_name
from utils.iter_counter import IterationCounter
from utils.visualizer import Visualizer


def main():
    # Get arguments
    args = Options().parse()

    # Set cuda environment
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu_ids)

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize metric
    metrics = StreamSegMetrics(args, args.num_classes)

    # Initialize Visualizer
    visualizer = Visualizer(args)

    # Initialize Iteration counter
    iter_counter = IterationCounter(args)

    # Define/Load model
    model_d = None
    model_d2 = None
    assert osp.exists(args.restore_from), f'Missing init model {args.restore_from}'
    if args.model == 'deeplabv2':
        model = deeplabv2_resnet(num_classes=args.num_classes, multi_level=args.multi_level)
        # Change normalization layer
        if args.seg_norm is not None:
            print("Changing norm with", args.seg_norm)
            change_normalization_layer(model, args.seg_norm)

        if args.is_train:
            saved_state_dict = torch.load(args.restore_from)
            if 'DeepLab_resnet_pretrained' in args.restore_from:
                new_params = model.state_dict().copy()
                for i in saved_state_dict:
                    i_parts = i.split('.')
                    if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                        new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                model.load_state_dict(new_params)
            else:
                model.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(f"Not yet supported {args.model}")
    print('Model Loaded')

    # Define discriminators
    model_d = PixDADiscriminator(args.num_classes)
    model_d2 = FCDiscriminator(args.num_classes)

    # Move model to cuda
    model = torch.nn.DataParallel(model)
    model = model.to(args.gpu_ids[0])
    model_d = torch.nn.DataParallel(model_d)
    model_d = model_d.to(args.gpu_ids[0])
    model_d2 = torch.nn.DataParallel(model_d2)
    model_d2 = model_d2.to(args.gpu_ids[0])

    # Set cudnn
    cudnn.benchmark = True
    cudnn.enabled = True

    # Define data loaders
    source_train_loader = None
    target_train_loader = None
    val_loader = None
    test_loader = None
    if args.is_train:
        # Define source train loader
        dataset_instance = find_dataset_using_name(args.source_dataset)
        source_dataset = dataset_instance(args=args,
                                          root=args.source_dataroot,
                                          mean=args.mean_prep,
                                          crop_size=args.crop_size,
                                          train=args.is_train,
                                          ignore_index=args.ignore_index)
        source_train_loader = data.DataLoader(source_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=True,
                                              pin_memory=True)

        # Define target train loader
        dataset_instance = find_dataset_using_name(args.target_dataset)
        target_dataset = dataset_instance(root=args.target_dataroot,
                                          mean=args.mean_prep,
                                          crop_size=args.crop_size,
                                          train=args.is_train,
                                          max_iters=args.max_iters*args.batch_size,
                                          ignore_index=args.ignore_index,
                                          num_shot=args.num_shots)
        target_train_loader = data.DataLoader(target_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              shuffle=True,
                                              drop_last=True,
                                              pin_memory=True)

        # Define val loader
        dataset_instance = find_dataset_using_name(args.target_dataset)
        val_dataset = dataset_instance(root=args.target_dataroot,
                                       mean=args.mean,
                                       crop_size=args.crop_size,
                                       train=False,
                                       ignore_index=args.ignore_index)
        val_loader = data.DataLoader(val_dataset,
                                     batch_size=args.batch_size_val,
                                     num_workers=args.num_workers,
                                     shuffle=False,
                                     pin_memory=True)
    else:
        # Define test loader
        dataset_instance = find_dataset_using_name(args.target_dataset)
        test_dataset = dataset_instance(root=args.target_dataroot,
                                        mean=args.mean,
                                        crop_size=args.crop_size,
                                        train=False,
                                        ignore_index=args.ignore_index)
        test_loader = data.DataLoader(test_dataset,
                                      batch_size=args.batch_size_val,
                                      num_workers=args.num_workers,
                                      shuffle=False,
                                      pin_memory=True)

    if args.is_train:
        # Launch training
        train_pixda(args, model, model_d, model_d2, source_train_loader, target_train_loader, val_loader, metrics,
                    iter_counter, visualizer)
    else:
        # Launch testing
        test(args, model, test_loader, metrics, visualizer)

    visualizer.close()


if __name__ == '__main__':
    main()


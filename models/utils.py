import torch.nn as nn
import torch
import os


def change_normalization_layer(model, norm_type):
    if norm_type == 'instance_norm':
        for child_name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(model, child_name, nn.InstanceNorm2d(num_features=child.num_features, track_running_stats=True, affine=True))
            else:
                change_normalization_layer(child, norm_type)
    if norm_type == 'group_norm':
        for child_name, child in model.named_children():
            if isinstance(child, nn.GroupNorm):
                setattr(model, child_name, nn.GroupNorm(num_groups=3, num_channels=child.num_features))
            else:
                change_normalization_layer(child, norm_type)


def freeze_model(args, model, exception):
    for parameter in model.module.parameters():
        parameter.requires_grad = False
    if exception == 'classifier':
        for parameter in model.module.layer6.parameters():
            parameter.requires_grad = True
        if args.multi_level:
            for parameter in model.module.layer5.parameters():
                parameter.requires_grad = True
    return model


def save_model(args, model, optimizer, scheduler, iter_counter):
    torch.save({
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'model_state': model.state_dict(),
        'total_steps_so_far': iter_counter.total_steps_so_far,
    }, os.path.join(args.checkpoints_dir, args.name, "models", "model_"+str(iter_counter.total_steps_so_far)+".pth"))


def load_model(args, model, optimizer=None, scheduler=None, so=False):
    if so:
        source_only_path = f".models/pretrained_models/{args.source_dataset}_source_only_model.pth"
        print(f'Resuming Model at: {source_only_path}')
        checkpoint = torch.load(source_only_path)
        model.load_state_dict(checkpoint['model_state'])
        return model
    else:
        print(f'Resuming Model at iter: {args.which_iter}')
        start_iter = args.which_iter
        state_dict_path = os.path.join(args.checkpoints_dir, args.name, "models", 'model_' + str(args.which_iter) + '.pth')
        checkpoint = torch.load(state_dict_path)
        model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and scheduler is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_iter = checkpoint['total_steps_so_far']
        return model, optimizer, scheduler, start_iter


def save_da_model(args,
                  model,
                  model_d1,
                  optimizer,
                  optimizer_d1,
                  scheduler,
                  scheduler_d1,
                  iter_counter,
                  model_d2=None,
                  optimizer_d2=None,
                  scheduler_d2=None,
                  ):
    if model_d2 is not None:
        torch.save({
            'optimizer_state': optimizer.state_dict(),
            'optimizer_d1_state': optimizer_d1.state_dict(),
            'optimizer_d2_state': optimizer_d2.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scheduler_d1_state': scheduler_d1.state_dict(),
            'scheduler_d2_state': scheduler_d2.state_dict(),
            'model_state': model.state_dict(),
            'model_d1_state': model_d1.state_dict(),
            'model_d2_state': model_d2.state_dict(),
            'total_steps_so_far': iter_counter.total_steps_so_far,
        }, os.path.join(args.checkpoints_dir, args.name, "models", "model_"+str(iter_counter.total_steps_so_far)+".pth"))
    else:
        torch.save({
            'optimizer_state': optimizer.state_dict(),
            'optimizer_d_state': optimizer_d1.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scheduler_d_state': scheduler_d1.state_dict(),
            'model_state': model.state_dict(),
            'model_d_state': model_d1.state_dict(),
            'total_steps_so_far': iter_counter.total_steps_so_far,
        }, os.path.join(args.checkpoints_dir, args.name, "models",
                        "model_" + str(iter_counter.total_steps_so_far) + ".pth"))


def load_da_model(args, model, model_d1, optimizer, optimizer_d1, scheduler, scheduler_d1, model_d2=None, optimizer_d2=None, scheduler_d2=None):
    print("Resuming Model")
    state_dict_path = os.path.join(args.checkpoints_dir, args.name, "models", 'model_' + str(args.which_iter) + '.pth')
    checkpoint = torch.load(state_dict_path)
    if model_d2 is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        optimizer_d1.load_state_dict(checkpoint['optimizer_d1_state'])
        optimizer_d2.load_state_dict(checkpoint['optimizer_d2_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scheduler_d1.load_state_dict(checkpoint['scheduler_d1_state'])
        scheduler_d2.load_state_dict(checkpoint['scheduler_d2_state'])
        model.load_state_dict(checkpoint['model_state'])
        model_d1.load_state_dict(checkpoint['model_d1_state'])
        model_d2.load_state_dict(checkpoint['model_d2_state'])
        start_iter = checkpoint['total_steps_so_far']
        return model, model_d1, model_d2, optimizer, optimizer_d1, optimizer_d2, scheduler, scheduler_d1, scheduler_d2, start_iter
    else:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        optimizer_d1.load_state_dict(checkpoint['optimizer_d_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scheduler_d1.load_state_dict(checkpoint['scheduler_d_state'])
        model.load_state_dict(checkpoint['model_state'])
        model_d1.load_state_dict(checkpoint['model_d_state'])
        start_iter = checkpoint['total_steps_so_far']
        return model, model_d1, optimizer, optimizer_d1, scheduler, scheduler_d1, start_iter

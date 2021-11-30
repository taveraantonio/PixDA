import torch
import torch.optim as optim
from tqdm import tqdm
from utils.scheduler import PolyLR
from utils.losses import CrossEntropy2d, FocalLoss, PixAdvLoss, get_target_tensor, KnowledgeDistillationLoss
from torch.nn import BCEWithLogitsLoss
from scripts.eval import validate
from models.utils import load_model, save_da_model, load_da_model
import torch.nn.functional as F
from dataset.utils import source_to_target, source_to_target_np
import torch.backends.cudnn as cudnn
from dataset.utils import find_dataset_using_name
from torch.utils import data
import copy


def train_pixda(args, model, model_d, model_d2, source_train_loader, target_train_loader, val_loader, metrics,
                iter_counter, visualizer):
    # Initialize variables
    cudnn.benchmark = True
    cudnn.enabled = True
    start_iter = 0
    parser_source_loss = 0.0
    parser_target_loss = 0.0
    parser_d_loss = 0.0
    discriminator_source_loss = 0.0
    discriminator_target_loss = 0.0
    d2_source_loss = 0.0
    d2_target_loss = 0.0
    metrics.reset()
    model = load_model(args, model, so=True)

    # Define optimizers
    optimizer = optim.SGD(model.module.optim_parameters(args.seg_lr),
                          lr=args.seg_lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    optimizer_d = optim.Adam(model_d.parameters(),
                             lr=args.d_lr,
                             betas=(args.beta1_d, args.beta2_d))
    optimizer_d2 = optim.Adam(model_d2.parameters(),
                              lr=args.d_lr,
                              betas=(args.beta1_d, args.beta2_d))

    # Define schedulers
    scheduler = PolyLR(optimizer,
                       max_iters=args.max_iters,
                       power=args.power)
    scheduler_d = PolyLR(optimizer_d,
                         max_iters=args.max_iters,
                         power=args.power)
    scheduler_d2 = PolyLR(optimizer_d2,
                          max_iters=args.max_iters,
                          power=args.power)

    # Define losses criterion
    if args.seg_loss == 'focal':
        criterion_seg = FocalLoss(num_class=args.num_classes,
                                  ignore_label=args.ignore_index)
    else:
        criterion_seg = CrossEntropy2d(ignore_label=args.ignore_index)
    criterion_d = BCEWithLogitsLoss()
    criterion_d1 = PixAdvLoss(args=args, ignore_index=args.ignore_index)

    # Resume model if continuing training
    if args.continue_train:
        model, model_d, model_d2, optimizer, optimizer_d, \
        optimizer_d2, scheduler, scheduler_d, scheduler_d2, start_iter = load_da_model(args,
                                                                                       model,
                                                                                       model_d,
                                                                                       optimizer,
                                                                                       optimizer_d,
                                                                                       scheduler,
                                                                                       scheduler_d,
                                                                                       model_d2,
                                                                                       optimizer_d2,
                                                                                       scheduler_d2)

    iter_counter.record_training_start(start_iter)

    # Start training
    source_train_loader_it = iter(source_train_loader)
    target_train_loader_it = iter(target_train_loader)
    iter_counter.record_training_start(start_iter)
    for i_iter in tqdm(iter_counter.training_steps()):
        # Set model to train
        model.train()
        model_d.train()
        model_d2.train()

        # Zero-grad the optimizers
        optimizer.zero_grad()
        optimizer_d.zero_grad()
        optimizer_d2.zero_grad()

        # Get source/target images and labels and move them to GPUs
        try:
            source_images, source_labels, _, _ = next(source_train_loader_it)
        except:
            iter_counter.record_one_epoch()
            visualizer.info('Validating model at step %d' % iter_counter.total_steps_so_far)
            # Validate and save model
            validate(args, model, val_loader, metrics, visualizer, iter_counter.total_steps_so_far)
            save_da_model(args, model, model_d, optimizer, optimizer_d, scheduler, scheduler_d, iter_counter, model_d2,
                          optimizer_d2, scheduler_d2)

            # Perform sample selection
            source_train_loader, counter = sample_selection(args, model, model_d2, source_train_loader,
                                                            target_train_loader, visualizer,
                                                            iter_counter.total_epochs())

            # If no more training sample for source stop training, otherwise continue
            if source_train_loader is not None:
                source_train_loader_it = iter(source_train_loader)
                source_images, source_labels, _, _ = next(source_train_loader_it)
            else:
                visualizer.info('Epoch end at %d' % iter_counter.total_epochs())
                break
        try:
            target_images, target_labels, _, name = next(target_train_loader_it)
        except:
            target_train_loader_it = iter(target_train_loader)
            target_images, target_labels, _, name = next(target_train_loader_it)

        # Image to image translation from source to target
        src_in_trg = source_images.clone()
        for cnt, (src_img, trg_img) in enumerate(zip(source_images, target_images)):
            src_in_trg[cnt, ...] = torch.from_numpy(source_to_target_np(src_img, trg_img, L=0.01))
        mean = torch.reshape(torch.from_numpy(args.mean), (1, 3, 1, 1))
        B, C, H, W = source_images.shape
        mean = mean.repeat(B, 1, H, W)
        source_images = src_in_trg.clone() - mean
        target_images = target_images - mean

        source_images, source_labels = source_images.to(args.gpu_ids[0], dtype=torch.float32), source_labels.to(
            args.gpu_ids[0], dtype=torch.long)
        target_images, target_labels = target_images.to(args.gpu_ids[0], dtype=torch.float32), target_labels.to(
            args.gpu_ids[0], dtype=torch.long)

        # Define Interpolation function
        interp = torch.nn.Upsample(size=(source_images.shape[2], source_images.shape[3]), mode='bilinear',
                                   align_corners=True).to(args.gpu_ids[0])

        # TRAIN SEGMENTATION MODEL
        # Don't accumulate gradients in discriminator
        for param in model_d.parameters():
            param.requires_grad = False

        # Train Source
        _, source_predictions = model(source_images)
        source_predictions = interp(source_predictions)
        loss_seg_source = criterion_seg(source_predictions, source_labels)
        loss_seg_source.backward()

        # Train Target
        _, target_predictions = model(target_images)
        target_predictions = interp(target_predictions)
        loss_seg_target = criterion_seg(target_predictions, target_labels)
        # And fool discriminator with PixAdv Loss
        d_output = model_d(F.softmax(target_predictions, dim=1))
        loss_pixadv = criterion_d1(target_predictions, d_output, target_labels, name=name[0])
        loss_target = loss_pixadv * args.lambda_adv + loss_seg_target
        loss_target.backward()

        # TRAIN DISCRIMINATORS
        for param in model_d.parameters():
            param.requires_grad = True

        source_predictions = source_predictions.detach()
        target_predictions = target_predictions.detach()

        d_output_source = model_d(F.softmax(source_predictions, dim=1))
        target_tensor = get_target_tensor(d_output_source, "source")
        source_d_loss = criterion_d(d_output_source, target_tensor.to(args.gpu_ids[0], dtype=torch.float)) / 2
        source_d_loss.backward()

        d_output_target = model_d(F.softmax(target_predictions, dim=1))
        target_tensor = get_target_tensor(d_output_target, "target")
        target_d_loss = criterion_d(d_output_target, target_tensor.to(args.gpu_ids[0], dtype=torch.float)) / 2
        target_d_loss.backward()

        # source training of the second discriminator for sample selection phase
        source_predictions = source_predictions.detach()
        target_predictions = target_predictions.detach()

        d2_output_source = model_d2(F.softmax(source_predictions, dim=1))
        source_d2_loss = criterion_d(d2_output_source, get_target_tensor(d2_output_source, "source").to(args.gpu_ids[0],
                                                                                                        dtype=torch.float)) / 2
        source_d2_loss.backward()

        # target training of the second discriminator for sample selection phase
        d2_output_target = model_d2(F.softmax(target_predictions, dim=1))
        target_d2_loss = criterion_d(d2_output_target, get_target_tensor(d2_output_target, "target").to(args.gpu_ids[0],
                                                                                                        dtype=torch.float)) / 2
        target_d2_loss.backward()

        # Update model
        optimizer.step()
        optimizer_d.step()
        optimizer_d2.step()
        scheduler.step()
        scheduler_d.step()
        scheduler_d2.step()

        # Update logging information
        parser_source_loss += loss_seg_source.item()
        parser_target_loss += loss_seg_target.item()
        parser_d_loss += loss_pixadv.item()
        discriminator_source_loss += source_d_loss.item()
        discriminator_target_loss += target_d_loss.item()
        d2_source_loss += source_d2_loss.item()
        d2_target_loss += target_d2_loss.item()

        # Print losses
        if iter_counter.needs_printing():
            # Print log and visualize on tensorboard
            visualizer.info(
                f'Parser source loss at iter {iter_counter.total_steps_so_far}: {parser_source_loss / args.print_freq}')
            visualizer.add_scalar('Parser_Source_Loss', parser_source_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Parser target loss at iter {iter_counter.total_steps_so_far}: {parser_target_loss / args.print_freq}')
            visualizer.add_scalar('Parser_Target_Loss', parser_target_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Parser discriminator loss at iter {iter_counter.total_steps_so_far}: {parser_d_loss / args.print_freq}')
            visualizer.add_scalar('Parser_Discriminator_Loss', parser_d_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Discriminator Source loss at iter {iter_counter.total_steps_so_far}: {discriminator_source_loss / args.print_freq}')
            visualizer.add_scalar('Discriminator_Source_Loss', discriminator_source_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Discriminator Target loss at iter {iter_counter.total_steps_so_far}: {discriminator_target_loss / args.print_freq}')
            visualizer.add_scalar('Discriminator_Target_Loss', discriminator_target_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'D2 Source loss at iter {iter_counter.total_steps_so_far}: {d2_source_loss / args.print_freq}')
            visualizer.add_scalar('D2_Source_Loss', d2_source_loss / args.print_freq, iter_counter.total_steps_so_far)
            visualizer.info(
                f'D2 Target loss at iter {iter_counter.total_steps_so_far}: {d2_target_loss / args.print_freq}')
            visualizer.add_scalar('D2_Target_Loss', d2_target_loss / args.print_freq, iter_counter.total_steps_so_far)

            parser_source_loss = 0.0
            parser_target_loss = 0.0
            parser_d_loss = 0.0
            discriminator_source_loss = 0.0
            discriminator_target_loss = 0.0
            d2_source_loss = 0.0
            d2_target_loss = 0.0

        iter_counter.record_one_iteration()

    iter_counter.record_training_end()
    visualizer.info('Starting fine tuning')

    # Last knowledge distillation and fine tuning phase
    train_kd(args, model, optimizer, scheduler, target_train_loader, iter_counter, visualizer)
    validate(args, model, val_loader, metrics, visualizer, iter_counter.total_steps_so_far)
    save_da_model(args, model, model_d, optimizer, optimizer_d, scheduler, scheduler_d, iter_counter, model_d2,
                  optimizer_d2, scheduler_d2)


def sample_selection(args, model, model_d2, source_train_loader, target_train_loader, visualizer, epoch=0):

    visualizer.info('Performing Sample selection at epoch: ' + str(epoch))
    # Set to eval mode
    model.eval()
    model_d2.eval()
    criterion_d = BCEWithLogitsLoss()
    ss_images = []

    # start selecting images
    target_train_loader_it = iter(target_train_loader)
    with torch.no_grad():
        for index, source_batch in tqdm(enumerate(source_train_loader)):

            try:
                target_images, _, _, _ = next(target_train_loader_it)
            except:
                target_train_loader_it = iter(target_train_loader)
                target_images, _, _, _ = next(target_train_loader_it)

            source_images, _, _, source_name = source_batch

            # Image to image translation from source to target
            src_in_trg = source_images.clone()
            for cnt, (src_img, trg_img) in enumerate(zip(source_images, target_images)):
                src_in_trg[cnt, ...] = torch.from_numpy(source_to_target_np(src_img, trg_img, L=0.01))
            # src_in_trg = source_to_target(source_images, target_images, L=0.01)
            mean = torch.reshape(torch.from_numpy(args.mean), (1, 3, 1, 1))
            B, C, H, W = source_images.shape
            mean = mean.repeat(B, 1, H, W)
            source_images = src_in_trg.clone() - mean
            images = source_images.to(args.gpu_ids[0], dtype=torch.float32)

            # Extract features from the model
            _, output = model(images)
            interp = torch.nn.Upsample(size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=True).to(args.gpu_ids[0])
            source_predictions = interp(output)
            d2_output_source = model_d2(F.softmax(source_predictions.detach(), dim=1))

            # Check prediction
            for idx, out in enumerate(d2_output_source):
                loss = criterion_d(out, get_target_tensor(out, "source").to(args.gpu_ids[0], dtype=torch.float))
                if loss.item() > args.ss_threshold:
                    ss_images.append(source_name[idx])

    # Recreate source train loader with the selected images
    del source_train_loader
    dataset_instance = find_dataset_using_name(args.source_dataset)
    source_dataset = dataset_instance(args=args,
                                      root=args.source_dataroot,
                                      mean=args.mean_pre,
                                      crop_size=args.crop_size,
                                      train=args.is_train,
                                      ignore_index=args.ignore_index,
                                      image_list=ss_images)
    source_train_loader = data.DataLoader(source_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=args.num_workers,
                                          drop_last=True,
                                          shuffle=True,
                                          pin_memory=True)
    args.ss_threshold += args.inc
    model.train()
    model_d2.train()
    return source_train_loader, len(ss_images)


def train_kd(args, model, optimizer, scheduler, target_train_loader, iter_counter, visualizer):

    # Initialize variables
    model_old = copy.deepcopy(model)

    # Define loss criterion
    if args.seg_loss == 'focal':
        criterion_seg = FocalLoss(num_class=args.num_classes, ignore_label=args.ignore_index)
    else:
        criterion_seg = CrossEntropy2d(ignore_label=args.ignore_index)
    criterion_kd = KnowledgeDistillationLoss(reduction='mean', alpha=args.alpha)

    target_train_loader_it = iter(target_train_loader)
    iter_counter.record_training_start(iter_counter.total_steps_so_far, True)
    for i_iter in tqdm(iter_counter.training_steps()):
        # Set model to train
        model.train()
        # Zero-grad the optimizer
        optimizer.zero_grad()

        # Get target images/labels and move them to GPUs
        try:
            images, labels, _, _ = next(target_train_loader_it)
        except:
            target_train_loader_it = iter(target_train_loader)
            images, labels, _, _ = next(target_train_loader_it)

        # Normalize images
        mean = torch.reshape(torch.from_numpy(args.mean), (1, 3, 1, 1))
        B, C, H, W = images.shape
        mean = mean.repeat(B, 1, H, W)
        images = images - mean
        images, labels = images.to(args.gpu_ids[0], dtype=torch.float32), labels.to(args.gpu_ids[0], dtype=torch.long)

        # Train
        interp = torch.nn.Upsample(size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=True).to(args.gpu_ids[0])
        with torch.no_grad():
            _, old_pred = model_old(images)
            old_pred = interp(old_pred)
        _, new_pred = model(images)
        new_pred = interp(new_pred)

        # Compute loss
        loss = criterion_seg(new_pred, labels) + args.lambda_kd * criterion_kd(new_pred, old_pred, labels)
        loss.backward()

        # Update model
        optimizer.step()
        scheduler.step()
        iter_counter.record_one_iteration()

    iter_counter.record_training_end()


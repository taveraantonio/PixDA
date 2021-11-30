import torch
from tqdm import tqdm
from utils.losses import CrossEntropy2d
from models.utils import load_model


def validate(args, model, val_loader, metrics, visualizer, val_iter):
    model.eval()
    metrics.reset()
    criterion_seg = CrossEntropy2d(ignore_label=args.ignore_index)

    results = {}
    val_loss = 0.0
    with torch.no_grad():
        for index, batch in tqdm(enumerate(val_loader), disable=False):
            image, label, _, name = batch
            _, pred_high = model(image.to(args.gpu_ids[0], dtype=torch.float32))

            interp = torch.nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True).to(args.gpu_ids[0])
            output = interp(pred_high)

            loss_seg = criterion_seg(output, label.to(args.gpu_ids[0], dtype=torch.long))
            val_loss += loss_seg.item()

            for img, lbl, out in zip(interp(image), label, output):
                visualizer.display_current_results([(img, lbl, out)], val_iter, 'Val')

            _, output = output.max(dim=1)
            output = output.cpu().numpy()
            label = label.cpu().numpy()
            metrics.update(label, output)

        visualizer.info(f'Validation loss at iter {val_iter}: {val_loss/len(val_loader)}')
        visualizer.add_scalar('Validation_Loss', val_loss/len(val_loader), val_iter)
        score = metrics.get_results()
        visualizer.add_figure("Val_Confusion_Matrix_Recall", score['Confusion Matrix'], step=val_iter)
        visualizer.add_figure("Val_Confusion_Matrix_Precision", score["Confusion Matrix Pred"], step=val_iter)
        results["Val_IoU"] = score['Class IoU']
        visualizer.add_results(results)
        visualizer.add_scalar('Validation_mIoU', score['Mean IoU'], val_iter)
        visualizer.info(metrics.to_str_print(score))
    metrics.reset()


def test(args, model, test_loader, metrics, visualizer):

    # Resume model and set to eval
    model, _, _, test_iter = load_model(args, model)
    model.eval()

    # Reset Metric and define variables for logging
    metrics.reset()
    results = {}

    # Start testing
    with torch.no_grad():
        for index, batch in tqdm(enumerate(test_loader), disable=False):
            images, labels, _, name = batch
            _, pred_high = model(images.to(args.gpu_ids[0], dtype=torch.float32))
            interp = torch.nn.Upsample(size=(labels.shape[1], labels.shape[2]), mode='bilinear', align_corners=True).to(
                args.gpu_ids[0])
            outputs = interp(pred_high)

            for img, lbl, out in zip(interp(images), labels, outputs):
                visualizer.display_current_results([(img, lbl, out)], val_iter, 'Val')

            _, outputs = outputs.max(dim=1)
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            metrics.update(labels, outputs)

        score = metrics.get_results()
        visualizer.add_figure("Test_Confusion_Matrix_Recall", score['Confusion Matrix'], step=test_iter)
        visualizer.add_figure("Test_Confusion_Matrix_Precision", score["Confusion Matrix Pred"], step=test_iter)
        results["Test_IoU"] = score['Class IoU']
        visualizer.add_results(results)
        visualizer.add_scalar('Test_mIoU', score['Mean IoU'], test_iter)
        visualizer.info(f'Test')
        visualizer.info(metrics.to_str_print(score))

    metrics.reset()


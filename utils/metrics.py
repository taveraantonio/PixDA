import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        pass

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def synch(self, device):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, args, n_classes):
        super().__init__()
        self.args = args
        self.n_classes = n_classes
        self.classes_name_19 = {
            0: "Road",
            1: "Sidewalk",
            2: "Building",
            3: "Wall",
            4: "Fence",
            5: "Pole",
            6: "TLight",
            7: "TSign",
            8: "Vegetation",
            9: "Terrain",
            10: "Sky",
            11: "Person",
            12: "Rider",
            13: "Car",
            14: "Truck",
            15: "Bus",
            16: "Train",
            17: "Motorcycle",
            18: "Bicycle"
        }
        self.classes_name_13 = {
            0: "Road",
            1: "Sidewalk",
            2: "Building",
            3: "TLight",
            4: "TSign",
            5: "Vegetation",
            6: "Sky",
            7: "Person",
            8: "Rider",
            9: "Car",
            10: "Bus",
            11: "Motorcycle",
            12: "Bicycle"
        }
        self.classes_name_16 = {
            0: "Road",
            1: "Sidewalk",
            2: "Building",
            3: "Wall",
            4: "Fence",
            5: "Pole",
            6: "TLight",
            7: "TSign",
            8: "Vegetation",
            9: "Terrain",
            10: "Sky",
            11: "Person",
            12: "Rider",
            13: "Vehicle",
            14: "Motorcycle",
            15: "Bicycle"
        }

        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_samples = 0

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        self.total_samples += len(label_trues)

    def to_str(self, results):
        string = "\n"
        ignore = ["Class IoU", "Class Acc", "Class Prec", "Class IoU 13",
                  "Confusion Matrix Pred", "Confusion Matrix", "Confusion Matrix Text"]
        for k, v in results.items():
            if k not in ignore:
                string += "%s: %f\n" % (k, v)

        string += 'Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Acc:\n'
        for k, v in results['Class Acc'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Prec:\n'
        for k, v in results['Class Prec'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        if self.args.source_dataset == 'synthia':
            string += '13 Class IoU:\n'
            for k, v in results['13 Class IoU'].items():
                string += "\tclass %d: %s\n" % (k, str(v))

        return string

    def to_str_print(self, results):
        string = "\n"
        ignore = ["Class IoU", "Class Acc", "Class Prec", "Class IoU 13",
                  "Confusion Matrix Pred", "Confusion Matrix", "Confusion Matrix Text"]
        for k, v in results.items():
            if k not in ignore:
                string += "%s: %f\n" % (k, v)

        string += 'Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tClass %s: %s\n" % (self.classes_name_19[k], str(v))

        if self.args.source_dataset == 'synthia':
            string += 'Class IoU 13:\n'
            for k, v in results['Class IoU 13'].items():
                string += "\tclass %s: %s\n" % (self.classes_name_13[k], str(v))

        string += "\n"
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - mean IU
        """
        EPS = 1e-9
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + EPS)
        acc_cls = np.mean(acc_cls_c[mask])
        precision_cls_c = diag / (hist.sum(axis=0) + EPS)
        precision_cls = np.mean(precision_cls_c)
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)
        # mean_iu = np.mean(iu[mask])
        mean_iu = round(np.nanmean(iu[mask]) * 100, 2)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        cls_iu = dict(zip(range(self.n_classes), [round(iu[i] * 100, 2) if m else "X" for i, m in enumerate(mask)]))
        cls_acc = dict(zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)]))
        cls_prec = dict(zip(range(self.n_classes), [precision_cls_c[i] if m else "X" for i, m in enumerate(mask)]))

        cls_iu_13 = {}
        for i, cls in enumerate([0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]):
            cls_iu_13[i] = cls_iu[cls]
        mean_iu_13 = round(np.nanmean(list(cls_iu_13.values())), 2)

        return {
            "Total samples": self.total_samples,
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "Mean Precision": precision_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            "Mean IoU 13": mean_iu_13,
            "Class IoU 13": cls_iu_13,
            "Class Acc": cls_acc,
            "Class Prec": cls_prec,
            "Confusion Matrix Text": self.confusion_matrix_to_text(),
            "Confusion Matrix": self.confusion_matrix_to_fig(),
            "Confusion Matrix Pred": self.confusion_matrix_to_fig(norm_gt=False)
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0

    def synch(self, device):
        # collect from multi-processes
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)
        samples = torch.tensor(self.total_samples).to(device)

        torch.distributed.reduce(confusion_matrix, dst=0)
        torch.distributed.reduce(samples, dst=0)

        if torch.distributed.get_rank() == 0:
            self.confusion_matrix = confusion_matrix.cpu().numpy()
            self.total_samples = samples.cpu().numpy()

    def confusion_matrix_to_fig(self, norm_gt=True):
        if norm_gt:
            div = (self.confusion_matrix.sum(axis=1) + 0.000001)[:, np.newaxis]
        else:
            div = (self.confusion_matrix.sum(axis=0) + 0.000001)[np.newaxis, :]
        cm = self.confusion_matrix.astype('float') / div
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title=f'Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        fig.tight_layout()
        return fig

    def confusion_matrix_to_text(self):
        string = []
        for i in range(self.n_classes):
            string.append(f"{i} : {self.confusion_matrix[i].tolist()}")
        return "\n" + "\n".join(string)


class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
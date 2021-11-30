import torch.nn as nn
import torch.nn.functional as F
import torch


def get_target_tensor(input_tensor, mode):
    # Source tensor = 0.0
    # Target tensor =  1.0
    source_tensor = torch.FloatTensor(1).fill_(0.0)
    target_tensor = torch.FloatTensor(1).fill_(1.0)
    source_tensor = source_tensor.expand_as(input_tensor)
    target_tensor = target_tensor.expand_as(input_tensor)
    if mode == 'source':
        return source_tensor
    elif mode == 'target':
        return target_tensor


class PixAdvLoss(nn.Module):

    def __init__(self, args, reduction='none', ignore_index=255, size_average=True):
        super(PixAdvLoss, self).__init__()
        self.args = args
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.criterion_d = nn.BCEWithLogitsLoss(reduction=self.reduction)

    def forward(self, parser_prediction, discriminator_pred, labels, fool="source", name=None):
        def get_balance(target_labels):
            balance = target_labels.clone()
            balance = balance.type(torch.FloatTensor)

            for i, label in enumerate(target_labels):
                classes = label.unique(sorted=True)
                pixels_count = torch.stack([(label == c).sum() for c in classes])
                tot_pixels = sum(pixels_count)
                for cl, px_count in zip(classes, pixels_count):
                    balance[i][balance[i] == float(cl.item())] = 1 - (px_count.item() / tot_pixels.item())
            return balance

        # Get fool tensor
        fool_tensor = get_target_tensor(discriminator_pred, fool).to(self.args.gpu_ids[0])
        # Compute discriminator loss (domain prediction) with fooling tensor
        fooling_loss = self.criterion_d(discriminator_pred, fool_tensor).squeeze()
        # Exploit target prediction and target labels to compute domain shift
        domain_shift = F.cross_entropy(parser_prediction, labels, reduction=self.reduction, ignore_index=self.ignore_index)
        # Exploit target ground truth to compute classes balances (for each pixel b = 1 - #pc/#totp)
        classes_balance = get_balance(labels).to(self.args.gpu_ids[0])
        # Compute final loss
        loss = fooling_loss * domain_shift * classes_balance

        # Return loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class KnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, gt, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])
        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)
        loss = (outputs * labels).mean(dim=1)

        if mask is not None:
            loss = loss * mask.float()
        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss
        return outputs


# Cross Entropy loss used for the semantic segmentation model
class CrossEntropy2d(nn.Module):

    def __init__(self, reduction='mean', ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
        Args:
        predict:(n, c, h, w)
        target:(n, h, w)
        weight (Tensor, optional): a manual rescaling weight given to each class.
                                   If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction=self.reduction)

        return loss


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=0.25, gamma=2.0, balance_index=2, size_average=True, ignore_label=255):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6
        self.ignore_label = ignore_label

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float,int)):
            assert 0<self.alpha<1.0, 'alpha should be in `(0,1)`)'
            assert balance_index >-1
            alpha = torch.ones((self.num_class))
            alpha *= 1-self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha,torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):
        logit = F.softmax(logit, dim=1)
        n, c, h, w = logit.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        logit = logit.transpose(1, 2).transpose(2, 3).contiguous()
        logit = logit[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        target = target.view(-1, 1)

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            alpha = self.alpha.to(logpt.device)
            alpha_class = alpha.gather(0,target.view(-1))
            logpt = alpha_class*logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCELoss(nn.Module):
    """
    CE over valid (non-padded) utterances.

    `weight` optionally re-weights classes (see utils.compute_class_weights).
    It lives on the loss object, never on the model, so model checkpoints are
    unaffected and load the same with or without it.
    """

    def __init__(self, weight=None):
        super(MaskedCELoss, self).__init__()
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, umask):
        umask = umask.view(-1, 1)
        target = target.view(-1, 1)
        pred = F.log_softmax(pred, 1)

        target_flat = (target * umask).squeeze().long()
        total = self.loss(pred * umask, target_flat)

        w = self.loss.weight
        if w is None:
            denom = torch.sum(umask)
        else:
            # Divide by the weights actually in play, not the sample count, so
            # the loss keeps its scale instead of growing with a batch's share
            # of rare classes -- otherwise CE would silently outweigh the
            # distillation terms it is summed with in train_student.py.
            denom = (w[target_flat] * umask.squeeze()).sum()

        return total / denom.clamp(min=1e-8)

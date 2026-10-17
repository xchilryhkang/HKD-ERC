import torch
import torch.nn as nn
import torch.nn.functional as F


## iemocap loss function: same with CE loss
class MaskedCELoss(nn.Module):
    def __init__(self):
        super(MaskedCELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target, umask):
        """
        Classification task loss (suitable for IEMOCAP), calculates NLLLoss for samples at valid positions.
        """

        umask = umask.view(-1, 1)
        target = target.view(-1, 1) 
        pred = F.log_softmax(pred, 1)

        loss = self.loss(pred * umask, (target * umask).squeeze().long()) / torch.sum(umask)
        return loss

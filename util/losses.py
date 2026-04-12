import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn


class UnchangedSimilarity(nn.Module):
    def __init__(self, reduction='mean', T=1.0):
        super(UnchangedSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)
        self.T = T

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()
        x1 = F.softmax(x1/self.T, dim=1)
        x2 = F.softmax(x2/self.T, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b*h*w, c])
        x2 = torch.reshape(x2, [b*h*w, c])

        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = torch.reshape(target, [b*h*w])

        loss = self.loss_f(x1, x2, target)
        return loss

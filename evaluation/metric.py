import torch
import numpy as np
import torch.nn.functional as F

from utils import is_tensor


def mean_iou(msk1, msk2, eps=1.0e-9):
    if not is_tensor(msk1):
        msk1 = torch.from_numpy(msk1).cuda()
    if not is_tensor(msk2):
        msk2 = torch.from_numpy(msk2).cuda()
    n = msk1.size(0)
    inter = msk1 * msk2
    union = ((msk1 + msk2) > 0.).float()
    miou = inter.view(n, -1).sum(-1) / (union.view(n, -1).sum(-1) + eps)
    return miou

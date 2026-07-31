import torch
import torch.nn.functional as F

from training.utils import *
from torch import nn

class DescriptorLoss(nn.Module):
    def __init__(self, inv_temp=20, dual_softmax_weight=5, heatmap_weight=1, detector_weighting=False, detector_weight_alpha=0.5):
        super().__init__()
        self.inv_temp = inv_temp
        self.dual_softmax_weight = dual_softmax_weight
        self.heatmap_weight = heatmap_weight
        self.detector_weighting = detector_weighting
        self.detector_weight_alpha = detector_weight_alpha

    def forward(self, m1, m2, h1, h2, pts1, pts2, detector_scores=None):
        loss_ds = dual_softmax_loss(
            m1,
            m2,
            temp=20,
            normalize=True,
            detector_scores=detector_scores,
            detector_weight_alpha=self.detector_weight_alpha,
        ) * self.dual_softmax_weight

        loss_h1, acc1 = heatmap_loss(h1, pts1)
        loss_h2, acc2 = heatmap_loss(h2, pts2)
        loss_h = (loss_h1 + loss_h2) / 2 * self.heatmap_weight

        if self.detector_weighting and detector_scores is not None:
            weight_mean = detector_scores.to(dtype=m1.dtype, device=m1.device).mean().clamp(min=1e-6)
            loss_h = loss_h * weight_mean

        acc_kp = 0.5 * (acc1 + acc2)

        return loss_ds, loss_h, acc_kp


def dual_softmax_loss(X, Y, temp=1, normalize=False, detector_scores=None, detector_weight_alpha=0.5):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    if normalize:
        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

    dist_mat = (X @ Y.t()) * temp

    P = dist_mat.softmax(dim=-2) * dist_mat.softmax(dim=-1)

    conf_gt = torch.eye(len(X), device=X.device)
    pos_mask = conf_gt == 1

    # focal loss
    alpha = 0.25
    gamma = 2
    pos_conf = P[pos_mask].clamp(min=1e-6, max=1 - 1e-6)
    loss_pos = -alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()

    if detector_scores is not None:
        if detector_scores.ndim != 1 or detector_scores.shape[0] != len(X):
            raise RuntimeError('Error: detector_scores must be a 1D tensor with one score per positive pair')
        detector_scores = detector_scores.to(device=X.device, dtype=X.dtype).clamp(min=0.0)
        cd_scores = pos_conf.clamp(min=1e-6, max=1.0)
        weights = detector_weight_alpha * detector_scores + (1.0 - detector_weight_alpha) * cd_scores
        weights = weights / weights.mean().clamp(min=1e-6)
        return (loss_pos * weights).mean()

    return loss_pos.mean()
    
def heatmap_loss(kpts, pts):
    C, H, W = kpts.shape

    with torch.no_grad():
        
        labels = torch.zeros((1, H, W), dtype=torch.long, device=kpts.device)
        labels[:, (pts[:,1]).long(), (pts[:,0]).long()] = 1
        
    kpts = kpts.view(-1)
    labels = labels.view(-1)
    
    BCE_loss = F.binary_cross_entropy(kpts, labels.float(), reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = 0.25 * (1 - pt) ** 2* BCE_loss
    
    with torch.no_grad():
        predictions = (kpts > 0.5)
        true_positives = ((predictions == 1) & (labels == 1)).sum().item()
        false_positives = ((predictions == 1) & (labels == 0)).sum().item()

        # Calculate Precision
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    return F_loss.mean(), precision

import torch
import torch.nn.functional as F

from training.utils import *
from torch import nn


def generate_candidate_patches(points_B, num_candidate_patches, patch_size, min_patch_points):
    """Generate square patches and return correspondence indices inside each patch."""
    if points_B.ndim != 2 or points_B.shape[-1] != 2 or points_B.numel() == 0:
        return []
    if num_candidate_patches <= 0 or patch_size <= 0:
        return []

    points = points_B.float()
    min_xy = points.min(dim=0).values
    max_xy = points.max(dim=0).values
    patch_size = float(patch_size)
    lower = min_xy
    upper = torch.maximum(min_xy, max_xy - patch_size)
    patches = []
    for _ in range(num_candidate_patches):
        top_left = lower + torch.rand(2, device=points.device) * (upper - lower)
        inside = (
            (points[:, 0] >= top_left[0])
            & (points[:, 0] < top_left[0] + patch_size)
            & (points[:, 1] >= top_left[1])
            & (points[:, 1] < top_left[1] + patch_size)
        )
        removed_indices = torch.where(inside)[0]
        if removed_indices.numel() >= min_patch_points:
            patches.append(removed_indices)
    return patches


def _valid_removed_indices(similarity, removed_indices):
    indices = torch.as_tensor(removed_indices, dtype=torch.long, device=similarity.device).flatten()
    indices = indices[(indices >= 0) & (indices < similarity.shape[0])]
    return torch.unique(indices)


def _remaining_indices(similarity, removed_indices):
    removed = _valid_removed_indices(similarity, removed_indices)
    keep = torch.ones(similarity.shape[1], dtype=torch.bool, device=similarity.device)
    keep[removed[removed < keep.numel()]] = False
    return removed, torch.where(keep)[0]


def compute_patch_confusion_score(similarity, removed_indices, margin_confusion, beta):
    removed, remaining = _remaining_indices(similarity, removed_indices)
    if removed.numel() == 0 or remaining.numel() == 0 or beta <= 0:
        return similarity.new_zeros(())
    positive = similarity[remaining, remaining]
    orphan = torch.logsumexp(beta * similarity[removed][:, remaining], dim=0) / beta
    return F.relu(margin_confusion + orphan - positive).mean()


def select_hard_patches(similarity, candidate_patches, num_hard_patches, margin_confusion, beta):
    if num_hard_patches <= 0:
        return []
    scored = [
        (compute_patch_confusion_score(similarity, patch, margin_confusion, beta).detach(), patch)
        for patch in candidate_patches
    ]
    scored.sort(key=lambda item: item[0].item(), reverse=True)
    return [patch for _, patch in scored[:num_hard_patches]]


def compute_orphan_loss(similarity, removed_indices, margin, beta):
    removed, remaining = _remaining_indices(similarity, removed_indices)
    if removed.numel() == 0 or remaining.numel() == 0 or beta <= 0:
        return similarity.new_zeros(())
    positive = similarity[remaining, remaining]
    orphan = torch.logsumexp(beta * similarity[removed][:, remaining], dim=0) / beta
    return F.relu(margin + orphan - positive).mean()


def compute_hard_patch_orphan_loss(
    similarity, candidate_patches, num_hard_patches, margin_confusion, margin, beta,
    return_metrics=False,
):
    hard_patches = select_hard_patches(
        similarity.detach(), candidate_patches, num_hard_patches, margin_confusion, beta
    )
    if not hard_patches:
        zero = similarity.sum() * 0.0
        result = zero, similarity.new_zeros(()), similarity.new_zeros(()), similarity.new_zeros(()), similarity.new_zeros(()), similarity.new_zeros(())
        return result if return_metrics else zero

    losses = [compute_orphan_loss(similarity, patch, margin, beta) for patch in hard_patches]
    valid_losses = [loss for loss in losses if loss is not None]
    if not valid_losses:
        zero = similarity.sum() * 0.0
        result = zero, similarity.new_zeros(()), similarity.new_zeros(()), similarity.new_zeros(()), similarity.new_zeros(()), similarity.new_zeros(())
        return result if return_metrics else zero
    orphan_loss = torch.stack(valid_losses).mean()
    scores = torch.stack([
        compute_patch_confusion_score(similarity.detach(), patch, margin_confusion, beta)
        for patch in hard_patches
    ])
    metric_values = []
    detached_similarity = similarity.detach()
    for patch in hard_patches:
        removed, remaining = _remaining_indices(detached_similarity, patch)
        if removed.numel() and remaining.numel():
            positive = detached_similarity[remaining, remaining]
            orphan = torch.logsumexp(beta * detached_similarity[removed][:, remaining], dim=0) / beta
            metric_values.append((positive.mean(), orphan.mean(), (orphan > positive).float().mean()))
    if metric_values:
        metric_tensor = torch.stack([torch.stack(values) for values in metric_values])
        mean_positive, mean_orphan, ratio = metric_tensor.mean(dim=0)
    else:
        mean_positive = mean_orphan = ratio = similarity.new_zeros(())
    result = orphan_loss, scores.mean(), scores.max(), mean_positive, mean_orphan, ratio
    return result if return_metrics else orphan_loss


class DescriptorLoss(nn.Module):
    def __init__(self, inv_temp=20, dual_softmax_weight=5, heatmap_weight=1,
                 lambda_orphan=0.0, margin=0.1, beta=20.0, num_candidate_patches=8,
                 num_hard_patches=2, patch_size=8.0, min_patch_points=10,
                 margin_confusion=0.1):
        super().__init__()
        self.inv_temp = inv_temp
        self.dual_softmax_weight = dual_softmax_weight
        self.heatmap_weight = heatmap_weight
        self.lambda_orphan = lambda_orphan
        self.margin = margin
        self.beta = beta
        self.num_candidate_patches = num_candidate_patches
        self.num_hard_patches = num_hard_patches
        self.patch_size = patch_size
        self.min_patch_points = min_patch_points
        self.margin_confusion = margin_confusion
    
    def forward(self, m1, m2, h1, h2, pts1, pts2, return_metrics=False):
        if m1.size() != m2.size() or m1.dim() != 2:
            raise RuntimeError('Error: descriptor matrices must have matching 2D shapes')
        normalized_m1 = F.normalize(m1, dim=-1)
        normalized_m2 = F.normalize(m2, dim=-1)
        similarity = (normalized_m1 @ normalized_m2.t()) * self.inv_temp
        loss_ds = dual_softmax_loss_from_similarity(similarity) * self.dual_softmax_weight
        if self.lambda_orphan > 0:
            candidate_patches = generate_candidate_patches(
                pts2, self.num_candidate_patches, self.patch_size, self.min_patch_points
            )
            orphan_loss, confusion, max_confusion, mean_positive, mean_orphan, ratio = compute_hard_patch_orphan_loss(
                similarity, candidate_patches, self.num_hard_patches, self.margin_confusion,
                self.margin, self.beta, return_metrics=True
            )
        else:
            orphan_loss = similarity.sum() * 0.0
            confusion = max_confusion = mean_positive = mean_orphan = ratio = similarity.new_zeros(())
                    
        loss_h1, acc1 = heatmap_loss(h1, pts1)
        loss_h2, acc2 = heatmap_loss(h2, pts2)
        loss_h = (loss_h1 + loss_h2) / 2 * self.heatmap_weight
        
        acc_kp = 0.5 * (acc1 + acc2)
        
        metrics = {
            'loss_dual_softmax': loss_ds.detach(),
            'loss_orphan': orphan_loss.detach(),
            'loss_total': (loss_ds + loss_h + self.lambda_orphan * orphan_loss).detach(),
            'mean_patch_confusion_score': confusion.detach(),
            'max_patch_confusion_score': max_confusion.detach(),
            'mean_positive_similarity': mean_positive.detach(),
            'mean_orphan_similarity': mean_orphan.detach(),
            'orphan_competition_ratio': ratio.detach(),
        }
        if return_metrics:
            return loss_ds + self.lambda_orphan * orphan_loss, loss_h, acc_kp, metrics
        return loss_ds, loss_h, acc_kp


def dual_softmax_loss_from_similarity(dist_mat):
    P = dist_mat.softmax(dim=-2) * dist_mat.softmax(dim=-1)
    pos_conf = P.diagonal()
    alpha = 0.25
    gamma = 2
    return (-alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()).mean()

def dual_softmax_loss(X, Y, temp = 1, normalize = False):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    if normalize:
        X = X/X.norm(dim=-1,keepdim=True)
        Y = Y/Y.norm(dim=-1,keepdim=True)
    
    dist_mat = (X @ Y.t()) * temp

    return dual_softmax_loss_from_similarity(dist_mat)
    
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

import torch
import torch.nn.functional as F

from training.utils import *
from torch import nn


@torch.no_grad()
def compute_geometry_difficulty(points0, points1, warp01_params, stride=1, eps=1e-6):
    """Compute a detached, per-point local viewpoint-change score.

    Both observations are used: viewing-ray angle is measured from the actual
    rays at ``points0`` and ``points1``, while scale and edge terms use depths
    sampled at both locations. This deliberately avoids treating the norm of
    a whole-image SE(3) transform as the stability target.
    """
    if points0.numel() == 0:
        return points0.new_zeros((0,))
    if points0.ndim != 2 or points1.ndim != 2 or points0.shape != points1.shape or points0.shape[-1] != 2:
        raise ValueError('points0 and points1 must be matching [N, 2] coordinate tensors')
    required = {
        'depth0', 'depth1', 'intrinsics0', 'intrinsics1', 'pose01',
        'bbox0', 'bbox1',
    }
    if not required.issubset(warp01_params):
        return points0.new_zeros((points0.shape[0],))
    device = points0.device
    depth0 = torch.as_tensor(warp01_params['depth0'], device=device, dtype=torch.float32)
    depth1 = torch.as_tensor(warp01_params['depth1'], device=device, dtype=torch.float32)
    intrinsics0 = torch.as_tensor(warp01_params['intrinsics0'], device=device, dtype=torch.float32)
    intrinsics1 = torch.as_tensor(warp01_params['intrinsics1'], device=device, dtype=torch.float32)
    pose01 = torch.as_tensor(warp01_params['pose01'], device=device, dtype=torch.float32)
    bbox0 = torch.as_tensor(warp01_params['bbox0'], device=device, dtype=torch.float32)
    bbox1 = torch.as_tensor(warp01_params['bbox1'], device=device, dtype=torch.float32)
    if (
        depth0.ndim != 2 or depth1.ndim != 2
        or intrinsics0.shape != (3, 3) or intrinsics1.shape != (3, 3)
        or pose01.shape != (4, 4)
    ):
        return points0.new_zeros((points0.shape[0],))

    pixel0 = points0.float() * float(stride)
    pixel1 = points1.float() * float(stride)

    def sample_map(value, pixels):
        height, width = value.shape
        grid = pixels.clone()
        grid[:, 0] = 2.0 * grid[:, 0] / max(width - 1, 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / max(height - 1, 1) - 1.0
        return F.grid_sample(
            value[None, None], grid.view(1, 1, -1, 2), mode='bilinear',
            padding_mode='zeros', align_corners=True,
        ).view(-1)

    sampled_depth0 = sample_map(depth0, pixel0).clamp_min(eps)
    sampled_depth1 = sample_map(depth1, pixel1).clamp_min(eps)

    global_pixel0 = pixel0 + bbox0[[1, 0]][None] + 0.5
    global_pixel1 = pixel1 + bbox1[[1, 0]][None] + 0.5
    homogeneous0 = torch.cat((global_pixel0, torch.ones_like(global_pixel0[:, :1])), dim=-1)
    homogeneous1 = torch.cat((global_pixel1, torch.ones_like(global_pixel1[:, :1])), dim=-1)
    point0 = (torch.linalg.inv(intrinsics0) @ homogeneous0.t()).t() * sampled_depth0[:, None]
    point1 = (torch.linalg.inv(intrinsics1) @ homogeneous1.t()).t() * sampled_depth1[:, None]
    rotation = pose01[:3, :3]

    # Convert the view-1 ray to camera-0 coordinates. Unlike the previous
    # source-only implementation, this depends directly on points1.
    ray0 = F.normalize(point0, dim=-1, eps=eps)
    ray1 = F.normalize((rotation.t() @ point1.t()).t(), dim=-1, eps=eps)
    view_angle = torch.acos((ray0 * ray1).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)) / torch.pi
    depth_scale = torch.log((sampled_depth1 + eps) / (sampled_depth0 + eps)).abs()

    dx = F.pad((depth0[:, 1:] - depth0[:, :-1]).abs(), (0, 1))
    dy = F.pad((depth0[1:, :] - depth0[:-1, :]).abs(), (0, 0, 0, 1))
    relative_edge0 = (dx + dy) / (depth0.abs() + eps)
    dx = F.pad((depth1[:, 1:] - depth1[:, :-1]).abs(), (0, 1))
    dy = F.pad((depth1[1:, :] - depth1[:-1, :]).abs(), (0, 0, 0, 1))
    relative_edge1 = (dx + dy) / (depth1.abs() + eps)
    edge = 0.5 * (sample_map(relative_edge0, pixel0) + sample_map(relative_edge1, pixel1))
    edge = edge.clamp(0.0, 2.0)
    return (view_angle + 0.25 * depth_scale.clamp(max=2.0) + 0.10 * edge).detach()


def compute_geometry_stability_loss(
    descriptors0, descriptors1, geometry_difficulty, base_tolerance=0.05,
    geometry_tolerance=0.20, tail_fraction=0.25, stability_logits0=None,
    stability_logits1=None,
):
    """Optimise excessive descriptor drift in geometry-comparable hard tails."""
    zero = descriptors0.sum() * 0.0
    if descriptors0.numel() == 0 or geometry_difficulty is None:
        return zero, zero, zero, zero, zero
    if geometry_difficulty.numel() != descriptors0.shape[0]:
        raise ValueError('geometry_difficulty must have one value per correspondence')
    descriptors0 = F.normalize(descriptors0, dim=-1)
    descriptors1 = F.normalize(descriptors1, dim=-1)
    drift = 1.0 - (descriptors0 * descriptors1).sum(dim=-1).clamp(-1.0, 1.0)
    difficulty = geometry_difficulty.detach().to(drift).clamp_min(0.0)
    tolerance = (float(base_tolerance) + float(geometry_tolerance) * difficulty).clamp(max=1.5)  # 不同的geometry_difficulty对应不同的容忍度
    excess = F.relu(drift - tolerance) / tolerance.clamp_min(1e-4)  # 超过容忍度的相对程度
    bin_edges = drift.new_tensor([0.15, 0.35, 0.70])
    bins = torch.bucketize(difficulty, bin_edges)
    hard_indices = []
    for bin_id in range(bin_edges.numel() + 1):  # 分成4个bin 选取每个bin中excess最大的百分之25的点作为hard_indices
        indices = torch.where(bins == bin_id)[0]
        if indices.numel() == 0:
            continue
        count = max(1, int(torch.ceil(indices.new_tensor(float(tail_fraction)).clamp(0.0, 1.0) * indices.numel()).item()))
        hard_indices.append(indices[torch.topk(excess[indices], k=count).indices])
    hard_indices = torch.cat(hard_indices) if hard_indices else torch.arange(drift.numel(), device=drift.device)
    drift_loss = F.smooth_l1_loss(excess[hard_indices], torch.zeros_like(excess[hard_indices])) # smooth_l1的特点是，小误差是平方，变化平缓，大误差是线性的形式，增长较快。

    score_loss = zero
    if stability_logits0 is not None and stability_logits1 is not None:
        logits = 0.5 * (stability_logits0.reshape(-1) + stability_logits1.reshape(-1))
        target = (drift <= tolerance).float().detach()
        if logits.numel() != target.numel():
            raise ValueError('stability logits must have one value per correspondence')
        score_loss = F.binary_cross_entropy_with_logits(logits, target)
    return drift_loss, score_loss, drift.mean(), tolerance.mean(), (drift <= tolerance).float().mean()
# drift loss 对应是 geometry-conditioned stability loss, 用于惩罚那些超过容忍度的点。
# score loss 对应是 stability score loss, 用于训练一个二分类器，预测哪些点是稳定的，哪些点是不稳定的。


def compute_relative_pose_difference(pose01, eps=1e-6):
    """Legacy global proxy retained only for compatibility with old scripts."""
    pose = torch.as_tensor(pose01)
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    if pose.ndim != 3 or pose.shape[-2:] != (4, 4):
        raise ValueError('pose01 must have shape [4, 4] or [B, 4, 4]')
    translation = torch.linalg.vector_norm(pose[:, :3, 3].float(), dim=-1)
    trace = pose[:, :3, :3].diagonal(dim1=-2, dim2=-1).sum(dim=-1).float()
    angle = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps))
    return 0.5 * (translation / (1.0 + translation)) + 0.5 * (angle / torch.pi)


def compute_stability_target(descriptors0, descriptors1, pose01=None, eps=1e-6):
    """Compatibility helper; the training pipeline uses local geometry instead."""
    descriptors0, descriptors1 = F.normalize(descriptors0, dim=-1), F.normalize(descriptors1, dim=-1)
    drift = 1.0 - (descriptors0 * descriptors1).sum(dim=-1).clamp(-1.0, 1.0)
    if pose01 is None:
        return (1.0 - drift).clamp(0.0, 1.0).detach()
    pose_change = compute_relative_pose_difference(pose01, eps=eps).mean().to(drift)
    return (1.0 / (1.0 + drift / pose_change.clamp_min(eps))).detach()


def weighted_pairwise_descriptor_loss(m1, m2, stability, temp=0.07, stability_weight=1.0, return_details=False):
    """Compatibility objective that focuses the legacy loss on unstable points."""
    if m1.dim() != 2 or m2.dim() != 2 or m1.shape[0] != m2.shape[0]:
        raise RuntimeError('m1 and m2 must be matching 2D tensors')
    m1, m2 = F.normalize(m1, dim=-1), F.normalize(m2, dim=-1)
    logits = (m1 @ m2.t()) / temp
    labels = torch.arange(logits.shape[0], device=logits.device)
    per_point = F.cross_entropy(logits, labels, reduction='none') + F.cross_entropy(logits.t(), labels, reduction='none')
    weights = 1.0 + float(stability_weight) * (1.0 - stability.to(logits).reshape(-1).clamp(0.0, 1.0))
    loss = (weights * per_point).mean()
    if not return_details:
        return loss
    return loss, {'unweighted_loss': per_point.mean().detach(), 'mean_stability_weight': weights.mean().detach()}


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
    return_metrics=False, return_details=False,
):
    hard_patches = select_hard_patches(
        similarity.detach(), candidate_patches, num_hard_patches, margin_confusion, beta
    )
    if not hard_patches:
        zero = similarity.sum() * 0.0
        result = zero, similarity.new_zeros(()), similarity.new_zeros(()), similarity.new_zeros(()), similarity.new_zeros(()), similarity.new_zeros(())
        details = {
            'active_orphan_fraction': similarity.new_zeros(()),
            'mean_positive_minus_orphan': similarity.new_zeros(()),
            'valid_patch_count': similarity.new_tensor(float(len(candidate_patches))),
            'hard_patch_count': similarity.new_zeros(()),
        }
        if return_details:
            return result + (details,)
        return result if return_metrics else zero

    losses = [compute_orphan_loss(similarity, patch, margin, beta) for patch in hard_patches]
    valid_losses = [loss for loss in losses if loss is not None]
    if not valid_losses:
        zero = similarity.sum() * 0.0
        result = zero, similarity.new_zeros(()), similarity.new_zeros(()), similarity.new_zeros(()), similarity.new_zeros(()), similarity.new_zeros(())
        details = {
            'active_orphan_fraction': similarity.new_zeros(()),
            'mean_positive_minus_orphan': similarity.new_zeros(()),
            'valid_patch_count': similarity.new_tensor(float(len(candidate_patches))),
            'hard_patch_count': similarity.new_tensor(float(len(hard_patches))),
        }
        if return_details:
            return result + (details,)
        return result if return_metrics else zero
    orphan_loss = torch.stack(valid_losses).mean()
    scores = torch.stack([
        compute_patch_confusion_score(similarity.detach(), patch, margin_confusion, beta)
        for patch in hard_patches
    ])
    metric_values = []
    active_values = []
    gap_values = []
    detached_similarity = similarity.detach()
    for patch in hard_patches:
        removed, remaining = _remaining_indices(detached_similarity, patch)
        if removed.numel() and remaining.numel():
            positive = detached_similarity[remaining, remaining]
            orphan = torch.logsumexp(beta * detached_similarity[removed][:, remaining], dim=0) / beta
            metric_values.append((positive.mean(), orphan.mean(), (orphan > positive).float().mean()))
            active_values.append((margin + orphan - positive > 0).float().mean())
            gap_values.append((positive - orphan).mean())
    if metric_values:
        metric_tensor = torch.stack([torch.stack(values) for values in metric_values])
        mean_positive, mean_orphan, ratio = metric_tensor.mean(dim=0)
    else:
        mean_positive = mean_orphan = ratio = similarity.new_zeros(())
    result = orphan_loss, scores.mean(), scores.max(), mean_positive, mean_orphan, ratio
    details = {
        'active_orphan_fraction': torch.stack(active_values).mean() if active_values else similarity.new_zeros(()),
        'mean_positive_minus_orphan': torch.stack(gap_values).mean() if gap_values else similarity.new_zeros(()),
        'valid_patch_count': similarity.new_tensor(float(len(candidate_patches))),
        'hard_patch_count': similarity.new_tensor(float(len(hard_patches))),
    }
    if return_details:
        return result + (details,)
    return result if return_metrics else orphan_loss


class DescriptorLoss(nn.Module):
    def __init__(self, inv_temp=20, dual_softmax_weight=5, heatmap_weight=1,
                 lambda_orphan=0.0, margin=0.1, beta=20.0, num_candidate_patches=8,
                 num_hard_patches=2, patch_size=8.0, min_patch_points=10,
                 margin_confusion=0.1, stability_loss_weight=0.0,
                 stability_score_weight=0.0, stability_base_tolerance=0.05,
                 stability_geometry_tolerance=0.20, stability_tail_fraction=0.25):
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
        self.stability_loss_weight = stability_loss_weight
        self.stability_score_weight = stability_score_weight
        self.stability_base_tolerance = stability_base_tolerance
        self.stability_geometry_tolerance = stability_geometry_tolerance
        self.stability_tail_fraction = stability_tail_fraction
    
    def forward(
        self, m1, m2, h1, h2, pts1, pts2, return_metrics=False,
        geometry_difficulty=None, stability_logits1=None, stability_logits2=None,
        pose01=None, stability1=None, stability2=None, stability_weight=None,
        stability_loss_weight=None,
    ):
        if m1.size() != m2.size() or m1.dim() != 2:
            raise RuntimeError('Error: descriptor matrices must have matching 2D shapes')
        normalized_m1 = F.normalize(m1, dim=-1)
        normalized_m2 = F.normalize(m2, dim=-1)
        similarity = (normalized_m1 @ normalized_m2.t()) * self.inv_temp
        loss_dual = dual_softmax_loss_from_similarity(similarity) * self.dual_softmax_weight
        loss_descriptor = loss_dual
        effective_stability_weight = self.stability_loss_weight if stability_loss_weight is None else stability_loss_weight
        if (effective_stability_weight > 0 or self.stability_score_weight > 0) and geometry_difficulty is not None:
            loss_stability, loss_stability_score, mean_drift, mean_tolerance, stable_fraction = compute_geometry_stability_loss(
                m1, m2, geometry_difficulty,
                base_tolerance=self.stability_base_tolerance,
                geometry_tolerance=self.stability_geometry_tolerance,
                tail_fraction=self.stability_tail_fraction,
                stability_logits0=stability_logits1,
                stability_logits1=stability_logits2,
            )
        else:
            loss_stability = similarity.sum() * 0.0
            loss_stability_score = similarity.sum() * 0.0
            mean_drift = mean_tolerance = stable_fraction = similarity.new_zeros(())
        if self.lambda_orphan > 0:
            candidate_patches = generate_candidate_patches(
                pts2, self.num_candidate_patches, self.patch_size, self.min_patch_points
            )
            orphan_loss, confusion, max_confusion, mean_positive, mean_orphan, ratio, orphan_details = compute_hard_patch_orphan_loss(
                similarity, candidate_patches, self.num_hard_patches, self.margin_confusion,
                self.margin, self.beta, return_metrics=True, return_details=True
            )
            orphan_gradient_norm = similarity.new_zeros(())
            if torch.is_grad_enabled() and similarity.requires_grad and orphan_loss.requires_grad:
                orphan_gradient = torch.autograd.grad(
                    orphan_loss, similarity, retain_graph=True, allow_unused=True
                )[0]
                if orphan_gradient is not None:
                    orphan_gradient_norm = orphan_gradient.detach().norm()
        else:
            orphan_loss = similarity.sum() * 0.0
            confusion = max_confusion = mean_positive = mean_orphan = ratio = similarity.new_zeros(())
            orphan_gradient_norm = similarity.new_zeros(())
            orphan_details = {
                'active_orphan_fraction': similarity.new_zeros(()),
                'mean_positive_minus_orphan': similarity.new_zeros(()),
                'valid_patch_count': similarity.new_zeros(()),
                'hard_patch_count': similarity.new_zeros(()),
            }
                    
        loss_h1, acc1 = heatmap_loss(h1, pts1)
        loss_h2, acc2 = heatmap_loss(h2, pts2)
        loss_h = (loss_h1 + loss_h2) / 2 * self.heatmap_weight
        
        acc_kp = 0.5 * (acc1 + acc2)
        
        metrics = {
            'loss_dual_softmax': loss_dual.detach(),
            'loss_descriptor': loss_descriptor.detach(),
            'loss_orphan': orphan_loss.detach(),
            'loss_total': (
                loss_descriptor + effective_stability_weight * loss_stability
                + self.stability_score_weight * loss_stability_score + loss_h
                + self.lambda_orphan * orphan_loss
            ).detach(),
            'loss_stability': loss_stability.detach(),
            'loss_stability_score': loss_stability_score.detach(),
            'mean_descriptor_drift': mean_drift.detach(),
            'mean_stability_tolerance': mean_tolerance.detach(),
            'geometry_stable_fraction': stable_fraction.detach(),
            'mean_patch_confusion_score': confusion.detach(),
            'max_patch_confusion_score': max_confusion.detach(),
            'mean_positive_similarity': mean_positive.detach(),
            'mean_orphan_similarity': mean_orphan.detach(),
            'orphan_competition_ratio': ratio.detach(),
            'active_orphan_fraction': orphan_details['active_orphan_fraction'].detach(),
            'mean_positive_minus_orphan': orphan_details['mean_positive_minus_orphan'].detach(),
            'valid_patch_count': orphan_details['valid_patch_count'].detach(),
            'hard_patch_count': orphan_details['hard_patch_count'].detach(),
            'orphan_similarity_grad_norm': orphan_gradient_norm.detach(),
        }
        loss_total_descriptor = (
            loss_descriptor + effective_stability_weight * loss_stability
            + self.stability_score_weight * loss_stability_score
            + self.lambda_orphan * orphan_loss
        )
        if return_metrics:
            return loss_total_descriptor, loss_h, acc_kp, metrics
        return loss_total_descriptor, loss_h, acc_kp


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

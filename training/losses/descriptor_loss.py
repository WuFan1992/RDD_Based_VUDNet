import torch
import torch.nn.functional as F

from training.utils import *
from torch import nn

def variance_invariance_disentanglement_loss(mu, variation, eps=1e-6):
    """Keep invariance and variation branches complementary instead of letting one branch absorb everything.

    The idea is to:
      1) minimize covariance between the two branch embeddings,
      2) keep both branches active (no branch collapse),
      3) prevent the network from transferring all information into a single branch.
    """
    if mu.dim() != 2 or variation.dim() != 2:
        raise RuntimeError('mu and variation must be 2D matrices [N, C]')
    if mu.size() != variation.size():
        raise RuntimeError('mu and variation shapes must match')

    mu = F.normalize(mu, dim=-1)
    variation = F.normalize(variation, dim=-1)

    mu_centered = mu - mu.mean(dim=0, keepdim=True)
    var_centered = variation - variation.mean(dim=0, keepdim=True)

    cov = (mu_centered.T @ var_centered) / max(1, mu.shape[0] - 1)
    scale_mu = mu_centered.norm(dim=0, keepdim=True).clamp_min(eps)
    scale_var = var_centered.norm(dim=0, keepdim=True).clamp_min(eps)
    corr = cov / (scale_mu * scale_var + eps)

    orthogonality = corr.abs().mean()

    mu_norm = mu_centered.pow(2).mean(dim=-1).sqrt().clamp_min(eps)
    var_norm = var_centered.pow(2).mean(dim=-1).sqrt().clamp_min(eps)
    branch_balance = (mu_norm.mean() - var_norm.mean()).abs() / (mu_norm.mean() + var_norm.mean() + eps)
    collapse_penalty = (1.0 / (mu_norm.mean() + eps) + 1.0 / (var_norm.mean() + eps)) * 0.5

    return orthogonality + 0.5 * branch_balance + 0.1 * collapse_penalty


class DescriptorLoss(nn.Module):
    def __init__(self, inv_temp=20, dual_softmax_weight=5, heatmap_weight=1,
                 stage=1, sigma_weight=0.0, reconstruction_weight=0.0,
                 disentangle_weight=0.5,
                 probabilistic_normalize_mu=True,
                 probabilistic_chunk_size=64,
                 matching_mode=None):
        super().__init__()
        self.inv_temp = inv_temp
        self.dual_softmax_weight = dual_softmax_weight
        self.heatmap_weight = heatmap_weight
        self.stage = stage
        self.sigma_weight = sigma_weight
        self.reconstruction_weight = reconstruction_weight
        self.disentangle_weight = float(disentangle_weight)
        self.probabilistic_normalize_mu = probabilistic_normalize_mu
        self.probabilistic_chunk_size = max(1, int(probabilistic_chunk_size))
        self.matching_mode = matching_mode or ("probabilistic" if stage >= 2 else "cosine")
        if self.matching_mode not in {"cosine", "probabilistic"}:
            raise ValueError("matching_mode must be 'cosine' or 'probabilistic'")
    
    def forward(self, m1, m2, h1, h2, pts1, pts2, logvar1=None, logvar2=None,
                reconstruction_loss=None, disentanglement_loss=None):
        if self.matching_mode == "probabilistic":
            loss_ds = probabilistic_dual_softmax_loss(
                m1,
                m2,
                logvar1,
                logvar2,
                temp=self.inv_temp,
                normalize_mu=self.probabilistic_normalize_mu,
                chunk_size=self.probabilistic_chunk_size,
            ) * self.dual_softmax_weight
        else:
            loss_ds = dual_softmax_loss(m1, m2, temp=self.inv_temp, normalize=True) * self.dual_softmax_weight
                    
        loss_h1, acc1 = heatmap_loss(h1, pts1)
        loss_h2, acc2 = heatmap_loss(h2, pts2)
        loss_h = (loss_h1 + loss_h2) / 2 * self.heatmap_weight
        
        acc_kp = 0.5 * (acc1 + acc2)
        
        loss_sigma = torch.zeros_like(loss_ds)
        raw_sigma = torch.zeros_like(loss_ds)
        if self.stage >= 2:
            raw_sigma = 0.5 * (variance_regularization(logvar1) + variance_regularization(logvar2))
            loss_sigma = raw_sigma * self.sigma_weight

        loss_rec = torch.zeros_like(loss_ds)
        if self.stage >= 4 and reconstruction_loss is not None:
            loss_rec = reconstruction_loss * self.reconstruction_weight

        loss_dis = torch.zeros_like(loss_ds)
        if self.stage >= 4 and disentanglement_loss is not None:
            loss_dis = disentanglement_loss * self.disentangle_weight

        total_loss = loss_ds + loss_sigma + loss_rec + loss_dis
        return total_loss, loss_h, acc_kp, loss_ds, loss_sigma, loss_rec, raw_sigma, loss_dis


def variance_regularization(logvar):
    variance = logvar.exp()
    return (variance - logvar - 1.0).mean()


def probabilistic_dual_softmax_loss(mu1, mu2, logvar1, logvar2, temp=1, normalize_mu=True, chunk_size=64):
    if mu1.size() != mu2.size() or logvar1.size() != mu1.size() or logvar2.size() != mu2.size():
        raise RuntimeError('Probabilistic descriptor shapes must match')
    if mu1.dim() != 2:
        raise RuntimeError('Probabilistic descriptors must be 2D matrices [N, D]')
    if normalize_mu:
        mu1 = F.normalize(mu1, dim=-1)
        mu2 = F.normalize(mu2, dim=-1)

    n, d = mu1.shape
    var1 = logvar1.exp()
    var2 = logvar2.exp()

    row_lse = mu1.new_empty(n)
    diag_logits = mu1.new_empty(n)
    col_lse = mu1.new_full((n,), -torch.inf)

    # Do not build [chunk, n, d] tensors in one shot.  The original formulation
    # broadcasts the full feature dimension, which creates very large autograd
    # intermediates and explodes memory as soon as the point count is no longer
    # capped.  Accumulate the per-feature contribution in smaller blocks instead.
    feature_block = max(1, min(32, d))

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        mu1_chunk = mu1[start:end]
        var1_chunk = var1[start:end]

        distance_chunk = mu1_chunk.new_zeros(end - start, n)
        for d_start in range(0, d, feature_block):
            d_end = min(d_start + feature_block, d)
            diff = mu1_chunk[:, None, d_start:d_end] - mu2[None, :, d_start:d_end]
            variance_sum = var1_chunk[:, None, d_start:d_end] + var2[None, :, d_start:d_end] + 1e-6
            block_term = (diff.square() / variance_sum).sum(dim=-1) / max(1, d_end - d_start)
            distance_chunk += block_term

        logits_chunk = -distance_chunk * temp

        row_lse[start:end] = torch.logsumexp(logits_chunk, dim=-1)
        col_lse = torch.logaddexp(col_lse, torch.logsumexp(logits_chunk, dim=0))
        local_ids = torch.arange(end - start, device=mu1.device)
        diag_logits[start:end] = logits_chunk[local_ids, start + local_ids]

    positive = torch.exp(2.0 * diag_logits - row_lse - col_lse).clamp_min(1e-6)
    return (-0.25 * (1.0 - positive).square() * positive.log()).mean()

def dual_softmax_loss(X, Y, temp = 1, normalize = False):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    if normalize:
        X = X/X.norm(dim=-1,keepdim=True)
        Y = Y/Y.norm(dim=-1,keepdim=True)
    
    dist_mat = (X @ Y.t()) * temp

    P = dist_mat.softmax(dim = -2) * dist_mat.softmax(dim= -1)
    
    conf_gt = torch.eye(len(X), device = X.device)
    pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
    
    conf_gt = torch.clamp(conf_gt, 1e-6, 1-1e-6)
    
    # focal loss
    alpha = 0.25
    gamma = 2
    pos_conf = P[pos_mask]
    loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()

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

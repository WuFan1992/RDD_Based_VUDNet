from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dataset.utils import EmptyTensorError, warp_se3

# 将feature map 的坐标归一化到[-1, 1]范围内，以便于grid_sample使用
def _normalize_feature_coords(coords: torch.Tensor, height: int, width: int, align_corners: bool) -> torch.Tensor:
    if align_corners:
        scale = coords.new_tensor([max(width - 1, 1), max(height - 1, 1)])
        return 2.0 * (coords / scale) - 1.0

    scale = coords.new_tensor([width, height])
    return (2.0 * coords + 1.0) / scale - 1.0

# 使用每个关键点的领域信息来对关键点的局部几何进行建模，预测每个关键点的仿射变换矩阵
class LocalGeometryHead(nn.Module):
    def __init__(self, input_dim: int, patch_size: int = 5, hidden_dim: int | None = None):
        super().__init__()
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd so the keypoint stays at the center of the neighborhood")
        self.input_dim = input_dim
        self.patch_size = patch_size
        hidden_dim = hidden_dim or max(input_dim // 2, 64)
        mid_dim = max(hidden_dim // 2, 32)
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(inplace=False),
            nn.Linear(mid_dim, 4),
        )
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, neighborhood: torch.Tensor) -> torch.Tensor:
        if neighborhood.numel() == 0:
            return neighborhood.new_zeros((*neighborhood.shape[:2], 2, 2))

        if neighborhood.ndim != 5:
            raise ValueError("LocalGeometryHead expects [B, N, C, H, W] local neighborhood patches")
        if neighborhood.shape[-1] != self.patch_size or neighborhood.shape[-2] != self.patch_size:
            raise ValueError(
                f"Expected a {self.patch_size}x{self.patch_size} neighborhood, got {neighborhood.shape[-2:]}")

        batch, points, channels, height, width = neighborhood.shape
        patch = neighborhood.reshape(batch * points, channels, height, width)
        feat = self.patch_encoder(patch).flatten(1)
        delta = self.regressor(feat).reshape(batch, points, 2, 2)

        eye = torch.eye(2, device=neighborhood.device, dtype=neighborhood.dtype).view(1, 1, 2, 2)
        return eye + delta

# LocalGeometryHead 预测周围局部的一个2x2的仿射变换，CanonicalSampler 把这个变换用到一个固定的canonical grid 上，然后从feature map中双线性采样
# 出一个几何自适应的局部patch
class CanonicalSampler(nn.Module):
    def __init__(
        self,
        grid_size: int = 16,
        radius: float = 8.0,
        *,
        align_corners: bool = False,
        padding_mode: str = "border",
        sample_chunk_size: int = 256,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.radius = float(radius)
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.sample_chunk_size = int(sample_chunk_size)

        ys = torch.linspace(-self.radius, self.radius, self.grid_size)
        xs = torch.linspace(-self.radius, self.radius, self.grid_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        self.register_buffer("canonical_grid", grid, persistent=False)
        
    # 给我一堆feature map 的坐标，我就帮你从feature map中采样出对应的特征点
    # feature_map: [B,C,H,W]     coords:  [B,N,M,2] 其中N是keypoints的数量，M是每个keypoint的采样点数量
    def _sample_points(self, feature_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature_map.shape
        batch_points = coords.shape[1]
        point_count = coords.shape[2]
        if batch_points == 0 or point_count == 0:
            return feature_map.new_zeros((batch, batch_points, channels, point_count))

        chunk_size = self.sample_chunk_size if self.sample_chunk_size > 0 else batch_points
        sampled_chunks = []
        for start in range(0, batch_points, chunk_size):
            end = min(start + chunk_size, batch_points)
            feat_flat = feature_map[:, None].expand(-1, end - start, -1, -1, -1).reshape(batch * (end - start), channels, height, width)
            coord_flat = coords[:, start:end].reshape(batch * (end - start), point_count, 2)
            grid = _normalize_feature_coords(coord_flat, height, width, self.align_corners).unsqueeze(-2)
            sampled = F.grid_sample(
                feat_flat,
                grid,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            sampled_chunks.append(sampled.squeeze(-1).reshape(batch, end - start, channels, point_count))

        return torch.cat(sampled_chunks, dim=1)  # [B,N,C,M]
    
    # 把一维的采样点坐标变成二维的patch
    def _sample_patch(self, feature_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        sampled = self._sample_points(feature_map, coords)
        batch, batch_points, channels, point_count = sampled.shape
        side = int(point_count ** 0.5)
        if side * side != point_count:
            raise ValueError(f"Expected a square number of points, got {point_count}")
        return sampled.reshape(batch, batch_points, channels, side, side) # [B,N,C,16,16]
    
    # 只取keypoints的中心点的特征
    def sample_center(self, feature_map: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature_map.shape
        batch_points = keypoints.shape[1]
        if batch_points == 0:
            return feature_map.new_zeros((batch, 0, channels))

        feat_flat = feature_map[:, None].expand(-1, batch_points, -1, -1, -1).reshape(batch * batch_points, channels, height, width)
        grid = _normalize_feature_coords(keypoints.reshape(batch * batch_points, 1, 2), height, width, self.align_corners).unsqueeze(-2)
        sampled = F.grid_sample(
            feat_flat,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return sampled.squeeze(-1).squeeze(-1).reshape(batch, batch_points, channels)

    def sample_neighborhood(self, feature_map: torch.Tensor, keypoints: torch.Tensor, neighborhood_size: int = 5) -> torch.Tensor:
        if neighborhood_size % 2 != 1:
            raise ValueError("neighborhood_size must be odd")

        batch, channels, height, width = feature_map.shape
        batch_points = keypoints.shape[1]
        if batch_points == 0:
            return feature_map.new_zeros((batch, 0, channels, neighborhood_size, neighborhood_size))

        half = neighborhood_size // 2
        offsets_y = torch.linspace(-half, half, neighborhood_size, device=feature_map.device, dtype=feature_map.dtype)
        offsets_x = torch.linspace(-half, half, neighborhood_size, device=feature_map.device, dtype=feature_map.dtype)
        offset_y, offset_x = torch.meshgrid(offsets_y, offsets_x, indexing="ij")
        offsets = torch.stack([offset_x, offset_y], dim=-1).reshape(-1, 2)

        coords = keypoints[:, :, None, :] + offsets[None, None, :, :]
        return self._sample_patch(feature_map, coords)

    # feature_map: [B, C, H, W], keypoints: [B, N, 2], affine: [B, N, 2, 2]
    def forward(self, feature_map: torch.Tensor, keypoints: torch.Tensor, affine: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = feature_map.shape
        batch_points = keypoints.shape[1]
        if batch_points == 0:
            empty_patch = feature_map.new_zeros((batch, 0, feature_map.shape[1], self.grid_size, self.grid_size))
            empty_mask = feature_map.new_zeros((batch, 0), dtype=torch.bool)
            empty_coords = feature_map.new_zeros((batch, 0, self.grid_size * self.grid_size, 2))
            return empty_patch, empty_mask, empty_coords

        grid = self.canonical_grid.to(device=feature_map.device, dtype=feature_map.dtype)
        offsets = torch.einsum("bnij,mj->bnmi", affine, grid)
        coords = keypoints[:, :, None, :] + offsets
        in_bounds = (
            (coords[..., 0] >= 0)
            & (coords[..., 0] <= width - 1)
            & (coords[..., 1] >= 0)
            & (coords[..., 1] <= height - 1)
        )
        valid_mask = in_bounds.all(dim=-1)
        patch = self._sample_patch(feature_map, coords)
        return patch, valid_mask, coords

# 把 CanonicalSampler 采样出来的，已经经过几何对齐的局部patch，经过一个残差卷积网络，得到每个关键点的描述子
class CanonicalDescriptorHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
    # [B, N, C, H, W]
    def forward(self, canonical_patch: torch.Tensor) -> torch.Tensor:
        batch, points, channels, height, width = canonical_patch.shape
        if points == 0:
            return canonical_patch.new_zeros((batch, 0, channels))

        patch = canonical_patch.reshape(batch * points, channels, height, width)
        patch = patch + self.residual(patch)
        patch = F.adaptive_avg_pool2d(patch, 1).flatten(1)
        return patch.reshape(batch, points, channels)


def compute_local_jacobian(
    keypoints_feat: torch.Tensor,
    warp_params: Dict[str, torch.Tensor],
    *,
    feature_stride: float,
    epsilon: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if keypoints_feat.numel() == 0:
        empty_j = keypoints_feat.new_zeros((0, 2, 2))
        empty_mask = keypoints_feat.new_zeros((0,), dtype=torch.bool)
        return empty_j, empty_mask

    keypoints_pix = keypoints_feat * feature_stride  # feature map 是原图的四分之一，为了恢复它在原图里的坐标，需要乘以feature_stride
    delta_x = keypoints_pix.new_tensor([epsilon, 0.0]) # delta x = (1, 0)
    delta_y = keypoints_pix.new_tensor([0.0, epsilon]) # delta y = (0, 1)

    def _run_warp(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            #它拿图像 0 中的 2D keypoints，通过 depth + camera intrinsics + 两帧之间的 SE(3) 相对位姿，把这些 2D 点“提升”为 3D 点 → 变换到相机 1 → 
            # 再投影回图像 1，最终得到 keypoint 在第二张图中的对应位置，同时剔除越界、无深度、遮挡等不可靠点。
            _, warped_tgt, ids_valid, _ = warp_se3(points, warp_params) 
        except EmptyTensorError:
            return points.new_zeros((0, 2)), points.new_zeros((0,), dtype=torch.long)
        return warped_tgt, ids_valid

    base_tgt, base_ids = _run_warp(keypoints_pix)
    x_tgt, x_ids = _run_warp(keypoints_pix + delta_x)
    y_tgt, y_ids = _run_warp(keypoints_pix + delta_y)

    num_points = keypoints_feat.shape[0]
    base_full = keypoints_pix.new_full((num_points, 2), float("nan"))  # 先创建 一个全是nan的tensor，后面再把有效的点放进去
    x_full = keypoints_pix.new_full((num_points, 2), float("nan"))
    y_full = keypoints_pix.new_full((num_points, 2), float("nan"))
    if base_ids.numel() > 0:
        base_full[base_ids] = base_tgt
    if x_ids.numel() > 0:
        x_full[x_ids] = x_tgt
    if y_ids.numel() > 0:
        y_full[y_ids] = y_tgt

    jacobian = torch.stack([x_full - base_full, y_full - base_full], dim=-1) / epsilon
    valid_mask = torch.isfinite(jacobian).all(dim=(-1, -2))
    return jacobian, valid_mask


def geometry_loss(
    affine0: torch.Tensor,
    affine1: torch.Tensor,
    jacobian_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    reduction: str = "squared_frobenius",
) -> torch.Tensor:
    if affine0.numel() == 0:
        return affine0.new_tensor(0.0)

    valid_mask = valid_mask & torch.isfinite(affine0).all(dim=(-1, -2)) & torch.isfinite(affine1).all(dim=(-1, -2)) & torch.isfinite(jacobian_gt).all(dim=(-1, -2))
    if not torch.any(valid_mask):
        return affine0.sum() * 0.0

    target = torch.matmul(jacobian_gt[valid_mask], affine0[valid_mask])
    diff = affine1[valid_mask] - target
    if reduction == "frobenius":
        return diff.norm(dim=(-1, -2)).mean()
    if reduction == "squared_frobenius":
        return diff.square().sum(dim=(-1, -2)).mean()
    raise ValueError(f"Unknown geometry loss reduction '{reduction}'")


def visualize_canonical_sampling(
    image0: torch.Tensor,
    image1: torch.Tensor,
    keypoints0: torch.Tensor,
    keypoints1: torch.Tensor,
    canonical_coords0: torch.Tensor,
    canonical_coords1: torch.Tensor,
    *,
    feature_stride: float = 1.0,
    point_index: int = 0,
    save_path: str | None = None,
):
    import matplotlib.pyplot as plt

    def _to_image(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        return tensor.detach().cpu()

    def _draw(ax, image, keypoint, coords, title: str):
        ax.imshow(image.clamp(0, 1))
        ax.scatter([keypoint[0]], [keypoint[1]], c="red", s=18)
        coords_pix = coords * feature_stride
        ax.scatter(coords_pix[:, 0], coords_pix[:, 1], s=4, c=torch.linspace(0, 1, coords_pix.shape[0]).cpu(), cmap="viridis", alpha=0.8)
        ax.set_title(title)
        ax.axis("off")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _draw(
        axes[0],
        _to_image(image0),
        keypoints0.detach().cpu()[point_index],
        canonical_coords0.detach().cpu()[point_index],
        "image0 canonical region",
    )
    _draw(
        axes[1],
        _to_image(image1),
        keypoints1.detach().cpu()[point_index],
        canonical_coords1.detach().cpu()[point_index],
        "image1 canonical region",
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
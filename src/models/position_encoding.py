import math
import torch
from torch import nn
from ..utils.misc import NestedTensor


def _deterministic_cumulative_sum(mask, dim):
    """Cumulative sum for boolean masks without CUDA's nondeterministic kernel."""
    result = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    length = mask.shape[dim]
    previous = None
    for index in range(length):
        current = mask.select(dim, index).to(torch.float32)
        if previous is not None:
            current = current + previous
        result.select(dim, index).copy_(current)
        previous = current
    return result


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        if mask is None:
            mask = torch.zeros_like(x[:, 0, :, :], dtype=torch.bool)
        not_mask = ~mask
        y_embed = _deterministic_cumulative_sum(not_mask, dim=1)
        x_embed = _deterministic_cumulative_sum(not_mask, dim=2)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
def build_position_encoding(config):
    N_steps = config['d_model'] // 2
    # TODO find a better way of exposing other arguments
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    return position_embedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.misc import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone

class BasicLayer(nn.Module):
	"""
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False),
									  nn.ReLU(inplace = False),
									)

	def forward(self, x):
	  return self.layer(x)

class RDD_Descriptor(nn.Module):
    def __init__(self, backbone, hidden_dim, num_feature_levels, invariant_dim=None, equivariant_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.invariant_dim = hidden_dim if invariant_dim is None else invariant_dim
        self.equivariant_dim = equivariant_dim

        if self.invariant_dim != self.hidden_dim:
            raise ValueError('invariant_dim must match hidden_dim to keep detector interface unchanged.')
        if self.equivariant_dim != 3:
            raise ValueError('equivariant_dim must be 3 to apply SE(3) pose constraints in training.')

        matchibility_hidden_dim = max(self.hidden_dim // 2, 64)
        matchibility_low_dim = max(matchibility_hidden_dim // 2, 32)

        # Shared encoder keeps the strong descriptor signal that the original RDD had.
        # We only use the split branch as a residual refinement, not as the entire descriptor.
        self.shared_head = nn.Sequential(
            BasicLayer(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1),
        )

        self.matchibility_head = nn.Sequential(
                                        BasicLayer(self.hidden_dim, matchibility_hidden_dim, 1, padding=0),
                                        BasicLayer(matchibility_hidden_dim, matchibility_low_dim, 1, padding=0),
                                        nn.Conv2d(matchibility_low_dim, 1, 1),
                                        nn.Sigmoid()
                                    )

        self.invariant_head = nn.Sequential(
                                        BasicLayer(self.hidden_dim, self.hidden_dim, 3, padding=1),
                                        nn.Conv2d(self.hidden_dim, self.invariant_dim, 1),
                                    )
        self.equivariant_head = nn.Sequential(
                                        BasicLayer(self.hidden_dim, self.hidden_dim // 2, 3, padding=1),
                                        nn.Conv2d(self.hidden_dim // 2, self.equivariant_dim, 1),
                                    )
        self.gate_head = nn.Sequential(
            BasicLayer(self.hidden_dim, self.hidden_dim // 2, 3, padding=1),
            nn.Conv2d(self.hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        self.backbone = backbone
        self.stride = backbone.strides[0]
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
            
    def forward(self, samples: NestedTensor, return_branches: bool = False):
        
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            if mask is None:
                mask = torch.zeros((src.shape[0], src.shape[-2], src.shape[-1]), dtype=torch.bool, device=src.device)
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                if m is None:
                    m = torch.zeros((src.shape[0], src.shape[-2], src.shape[-1]), dtype=torch.bool, device=src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        
        feats = srcs

        final_feature = feats[0]
        for feat in feats[1:]:
            final_feature = final_feature + F.interpolate(feat, size=final_feature.shape[-2:], mode='bilinear', align_corners=False)

        shared_map = self.shared_head(final_feature)
        invariant_map = self.invariant_head(final_feature)
        equivariant_map = self.equivariant_head(final_feature)
        gate = self.gate_head(final_feature)

        # Residual split: keep a strong shared descriptor and let the invariant branch act as a
        # learned correction / modulation, instead of replacing the main descriptor.
        descriptor_map = shared_map + gate * invariant_map
        matchibility = self.matchibility_head(descriptor_map)

        if return_branches:
            return descriptor_map, matchibility, {
                'equivariant_map': equivariant_map,
                'invariant_map': invariant_map,
                'shared_map': shared_map,
                'gate': gate,
            }

        return descriptor_map, matchibility
    
    
def build_descriptor(config):
    backbone = build_backbone(config)
    return RDD_Descriptor(
        backbone,
        hidden_dim=config['d_model'],
        num_feature_levels=config['num_feature_levels'],
        invariant_dim=config.get('invariant_dim', config['d_model']),
        equivariant_dim=config.get('equivariant_dim', 3),
    )

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
    def __init__(self, backbone, hidden_dim, num_feature_levels, descriptor_stage=1, logvar_min=-6.0, logvar_max=6.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.descriptor_stage = int(descriptor_stage)
        if self.descriptor_stage not in {1, 2, 3, 4}:
            raise ValueError("descriptor_stage must be one of 1, 2, 3, or 4")
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.mu_head = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.logvar_head = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.variation_head = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )

        matchibility_hidden_dim = max(self.hidden_dim // 2, 64)
        matchibility_low_dim = max(matchibility_hidden_dim // 2, 32)
        
        self.matchibility_head = nn.Sequential(
										BasicLayer(self.hidden_dim, matchibility_hidden_dim, 1, padding=0),
										BasicLayer(matchibility_hidden_dim, matchibility_low_dim, 1, padding=0),
										nn.Conv2d(matchibility_low_dim, 1, 1),
										nn.Sigmoid()
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
            
    def forward(self, samples: NestedTensor):
        
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
        
        mu = self.mu_head(final_feature)
        logvar = self.logvar_head(final_feature).clamp(self.logvar_min, self.logvar_max)
        variation = self.variation_head(final_feature)
        reconstruction = self.reconstruction_head(torch.cat((mu, variation), dim=1))
        matchibility = self.matchibility_head(mu)

        return mu, matchibility, logvar, variation, reconstruction, final_feature
    
    
def build_descriptor(config):
    backbone = build_backbone(config)
    return RDD_Descriptor(
        backbone,
        hidden_dim=config['d_model'],
        num_feature_levels=config['num_feature_levels'],
        descriptor_stage=config.get('descriptor_stage', 1),
        logvar_min=config.get('logvar_min', -6.0),
        logvar_max=config.get('logvar_max', 6.0),
    )

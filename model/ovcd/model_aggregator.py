import torch.nn as nn

from .model_agg_spatial import SptialFusionBlock


class EffAggregatorLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, drop_path=0., sr_ratio=2):
        super().__init__()
        self.spatial_fusion = SptialFusionBlock(hidden_dim, num_heads, drop_path=drop_path, sr_ratio=sr_ratio)

    def forward(self, diff, featA, featB):

        diff = self.spatial_fusion(diff, featA, featB)

        return diff

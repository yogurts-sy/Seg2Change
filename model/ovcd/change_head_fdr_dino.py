import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ovcd.blocks import FeatureFusionBlock

from model.ovcd.change_blocks import MDFM

from model.ovcd.model_aggregator import EffAggregatorLayer


class CHead_FPN_FDR(nn.Module):
    def __init__(self, in_channels, hidden_dim, nclass, multi=True):
        super().__init__()

        # FMM
        self.adapterA = nn.ModuleList([
            nn.Conv2d(in_channels, 48, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels, 80, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels, 96, kernel_size=1, stride=1, padding=0),
        ])

        self.adapterB = nn.ModuleList([
            nn.Conv2d(in_channels, 48, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels, 80, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels, 96, kernel_size=1, stride=1, padding=0),
        ])

        # BDFM
        self.diff_fdr0 = MDFM(48, 48)
        self.diff_fdr1 = MDFM(64, 64)
        self.diff_fdr2 = MDFM(80, 80)
        self.diff_fdr3 = MDFM(96, 96)

        # EDQA
        self.aggregator0 = EffAggregatorLayer(48)
        self.aggregator1 = EffAggregatorLayer(64)
        self.aggregator2 = EffAggregatorLayer(80)
        self.aggregator3 = EffAggregatorLayer(96)

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=48,
                out_channels=48,
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        self.layer0_rn = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, groups=1)
        self.layer1_rn = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1)
        self.layer2_rn = nn.Conv2d(80, 64, kernel_size=3, stride=1, padding=1, groups=1)
        self.layer3_rn = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, groups=1)

        self.refinenet0 = FeatureFusionBlock(hidden_dim, nn.ReLU(False))
        self.refinenet1 = FeatureFusionBlock(hidden_dim, nn.ReLU(False))
        self.refinenet2 = FeatureFusionBlock(hidden_dim, nn.ReLU(False))
        self.refinenet3 = FeatureFusionBlock(hidden_dim, nn.ReLU(False))

        self.class_embed = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, nclass, kernel_size=1, stride=1, padding=0)
        )

        self.multi = multi

        if self.multi:
            self.class_embed3 = nn.Conv2d(hidden_dim, nclass, kernel_size=1, stride=1, padding=0)
            self.class_embed2 = nn.Conv2d(hidden_dim, nclass, kernel_size=1, stride=1, padding=0)
            self.class_embed1 = nn.Conv2d(hidden_dim, nclass, kernel_size=1, stride=1, padding=0)
            self.class_embedA = nn.Conv2d(hidden_dim, nclass, kernel_size=1, stride=1, padding=0)
            self.class_embedB = nn.Conv2d(hidden_dim, nclass, kernel_size=1, stride=1, padding=0)

    def upsampler(self, feat0, feat1, feat2, feat3, get_multi=False):

        feat0 = self.layer0_rn(feat0)
        feat1 = self.layer1_rn(feat1)
        feat2 = self.layer2_rn(feat2)
        feat3 = self.layer3_rn(feat3)

        path3 = self.refinenet3(feat3, size=feat2.shape[2:])
        path2 = self.refinenet2(path3, feat2, size=feat1.shape[2:])
        path1 = self.refinenet1(path2, feat1, size=feat0.shape[2:])
        output = self.refinenet0(path1, feat0)

        if get_multi:
            return output, path1, path2, path3

        return output


    def forward(self, dino_enc_feats_A, dino_enc_feats_B):
        H, W = 504, 504
        patch_h, patch_w = H // 14, W // 14
        feats_A, feats_B = [], []

        # Adaptor for DINO features
        for i in range(len(dino_enc_feats_A)):
            fa, fb = dino_enc_feats_A[i], dino_enc_feats_B[i]
            fa = fa.permute(0, 2, 1).reshape((fa.shape[0], fa.shape[-1], patch_h, patch_w))
            fb = fb.permute(0, 2, 1).reshape((fb.shape[0], fb.shape[-1], patch_h, patch_w))

            fa = self.adapterA[i](fa)
            fb = self.adapterB[i](fb)

            feats_A.append(self.resize_layers[i](fa))
            feats_B.append(self.resize_layers[i](fb))

        # Feature Difference Extraction
        feat0 = self.diff_fdr0(feats_A[0], feats_B[0])
        feat1 = self.diff_fdr1(feats_A[1], feats_B[1])
        feat2 = self.diff_fdr2(feats_A[2], feats_B[2])
        feat3 = self.diff_fdr3(feats_A[3], feats_B[3])

        # Feature Difference Refinement
        feat0 = self.aggregator0(feat0, feats_A[0], feats_B[0])
        feat1 = self.aggregator1(feat1, feats_A[1], feats_B[1])
        feat2 = self.aggregator2(feat2, feats_A[2], feats_B[2])
        feat3 = self.aggregator3(feat3, feats_A[3], feats_B[3])

        output, path1, path2, path3 = self.upsampler(feat0, feat1, feat2, feat3, get_multi=True)
        outputA = self.upsampler(feats_A[0], feats_A[1], feats_A[2], feats_A[3], get_multi=False)
        outputB = self.upsampler(feats_B[0], feats_B[1], feats_B[2], feats_B[3], get_multi=False)

        output = self.class_embed(output)
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)

        output1 = self.class_embed1(path1)
        output1 = F.interpolate(output1, size=(H, W), mode='bilinear', align_corners=False)

        output2 = self.class_embed2(path2)
        output2 = F.interpolate(output2, size=(H, W), mode='bilinear', align_corners=False)

        output3 = self.class_embed3(path3)
        output3 = F.interpolate(output3, size=(H, W), mode='bilinear', align_corners=False)

        outputA = self.class_embedA(outputA)
        outputA = F.interpolate(outputA, size=(H, W), mode='bilinear', align_corners=False)

        outputB = self.class_embedB(outputB)
        outputB = F.interpolate(outputB, size=(H, W), mode='bilinear', align_corners=False)

        return output, output1, output2, output3, outputA, outputB
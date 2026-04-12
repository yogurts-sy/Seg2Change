import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class MDFM(nn.Module):
    def __init__(self, in_d, out_d):
        super(MDFM, self).__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.conv_sub = nn.Conv2d(self.in_d, self.in_d, 3, padding=1, bias=False)

        self.conv_diff_enh1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_diff_enh2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, self.in_d),
            nn.ReLU(inplace=True)
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, self.in_d),
            nn.ReLU(inplace=True)
        )

        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2): # current best

        x_sub = torch.abs(x1 - x2)
        x_att = torch.sigmoid(self.conv_sub(x_sub))

        x1 = self.conv_diff_enh1(x1.mul(x_att) + x1)
        x2 = self.conv_diff_enh2(x2.mul(x_att) + x2)
        # fusion
        x_f = torch.cat([x1, x2], dim=1)
        x_f = self.conv_cat(x_f)

        x = x_f * x_att
        out = self.conv_dr(x)

        return out
    
    def forward1(self, x1, x2):
        b, c, h, w = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3]
        x_sub = torch.abs(x1 - x2)
        x_att = torch.sigmoid(self.conv_sub(x_sub))
        x_cat = self.x_concat(torch.cat([x1, x2], dim = 1))
        x_att = self.bam(x_sub, x_cat)
        x1 = (x1 * x_att) 
        x2 = (x2 * x_att) 
        x_f = torch.stack((x1, x2), dim=2)
        x_f = torch.reshape(x_f, (b, -1, h, w))
        x_f = self.convmix(x_f)
        x_f = x_f * x_att
        out = self.conv_dr(x_f)
        return out
    
    def forward_3(self, x1, x2):
        b, c, h, w = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3]
        x_sub = torch.abs(x1 - x2)
        x_att = torch.sigmoid(self.conv_sub(x_sub))

        x1 = self.conv_diff_enh1(x1.mul(x_att) + x1)
        x2 = self.conv_diff_enh2(x2.mul(x_att) + x2)
        # fusion
        x_f = torch.cat([x1, x2], dim=1)
        x_f = self.conv_cat(x_f)

        x = x_f * x_att
        out = self.conv_dr(x)

        return out
    
    def forward_4(self, x1, x2):
        x_sub1 = torch.abs(x1 - x2)
        x_att1 = torch.sigmoid(self.conv_sub1(x_sub1))

        x1 = self.conv_diff_enh1(x1.mul(x_att1) + x1)
        x2 = self.conv_diff_enh2(x2.mul(x_att1) + x2)

        x_sub2 = torch.abs(x1 - x2)
        x_att2 = torch.sigmoid(self.conv_sub2(x_sub2))

        x1_muled = self.conv_diff_enh3(x1.mul(x_att2))
        x2_muled = self.conv_diff_enh4(x2.mul(x_att2))

        x_f = torch.cat([x1_muled, x2_muled], dim=1)
        x_f = self.conv_cat(x_f)

        out = self.conv_dr(x_f)

        return out


class MDFM_QKV(nn.Module):
    def __init__(self, in_d, out_d):
        super(MDFM_QKV, self).__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.conv_sub = nn.Conv2d(self.in_d, self.in_d // 4, 3, padding=1, bias=False)
        self.key = nn.Conv2d(self.in_d, self.in_d // 4, 3, padding=1, bias=False)
        self.value = nn.Conv2d(self.in_d, self.in_d // 4, 3, padding=1, bias=False)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, self.in_d),
            nn.ReLU(inplace=True)
        )

        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, self.out_d),
            nn.ReLU(inplace=True)
        )

        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x1, x2):
        b, c, h, w = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3]
        x_sub = torch.abs(x1 - x2)
        query = self.conv_sub(x_sub)       # [1, 32, 288, 288]

        x_f = torch.cat([x1, x2], dim=1)
        x_f = self.conv_cat(x_f)

        q = query.view(b, -1, h * w).permute(0, 2, 1)  # [1, 82944, 32]
        k = self.key(x_f).view(b, -1, h * w)           # [1, 32, 82944]
        v = self.value(x_f).view(b, -1, h * w)         # [1, 32, 82944]

        attn_matrix1 = torch.bmm(q, k)
        attn_matrix1 = torch.softmax(attn_matrix1)

        out = torch.bmm(v, attn_matrix1.permute(0, 2, 1)) 
        out = out.view(*x1.shape)

        out = self.conv_dr(out)

        change_feat = query + self.gamma * out

        return change_feat
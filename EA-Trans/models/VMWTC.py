
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.wtcnn3D import WTConv3d


class ChannelSELayer3D(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class SpatialSELayer3D(nn.Module):
    def __init__(self, channel):
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(channel, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply convolution to get a single channel output
        x = self.conv(x)
        # Apply sigmoid to get weights in the range [0, 1]
        x = self.sigmoid(x)
        return x


class VMWTC(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., kernel_size=1,
                 with_bn=True, dim_head=64):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.scale = dim_head ** -0.5

        # Pointwise
        self.conv1 = nn.Conv3d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)

        # Depthwise dilated
        self.conv2 = nn.Conv3d(
            hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, dilation=2, groups=hidden_features)

        # Depthwise dilated convolutions with variable dilation rates
        self.dilated_convs = nn.ModuleList([
            WTConv3d(in_channels = hidden_features, out_channels=hidden_features, kernel_size=kernel, wt_levels=3)
            for kernel in [1, 3, 5]  # Example dilation rates: 1, 2, 4
        ])

        # Channel SE Block
        self.cSE = ChannelSELayer3D(hidden_features)

        # Pointwise
        self.conv3 = nn.Conv3d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()

        self.bn = nn.ModuleList([nn.BatchNorm3d(hidden_features) for _ in range(len(self.dilated_convs))])
        self.bn1 = nn.BatchNorm3d(hidden_features)
        self.bn2 = nn.BatchNorm3d(hidden_features)
        self.bn3 = nn.BatchNorm3d(out_features)

        # Spatial SE Block
        self.sSE = SpatialSELayer3D(hidden_features)

        # The reduction ratio is always set to 4
        self.squeeze = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.compress = nn.Linear(in_features, in_features // 4)
        self.excitation = nn.Linear(in_features // 4, in_features)

    def forward(self, x):
        B, N, C = x.size()
        D = 5
        H = 8
        W = 8
        #cls_token, tokens = torch.split(x, [1, N - 1], dim=1)
        x = x.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        shortcut = x
        #print(x.shape)
        dilated_features = []
        for conv, bn in zip(self.dilated_convs, self.bn):
            dilated_features.append(self.act(bn(conv(x))))
        x = sum(dilated_features) / len(dilated_features)
        #print(dilated_features.shape)

        x = shortcut + x

        # Apply spatial SE block
        spatial_attention = self.sSE(x)
        x = x * spatial_attention

        # Channel SE Block
        x = self.cSE(x)

        x = self.conv3(x)
        x = self.bn3(x)

        out = x.flatten(2).permute(0, 2, 1)

        return out


"""# Example usage
dmssce= VMWTC(in_features=192)
x = torch.randn(32, 320, 192)
out = dmssce(x)
print(out.shape)"""
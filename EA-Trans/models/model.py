import torch
from torch import nn, einsum

from models.models.VMWTC import VMWTC
from models.WTConv import WTConv2d
from models.SASFA import SASFA
from models.edgedwcnn3D import EADWConv3d
from models.wtcnn3D import WTConv3d
from models.est import est
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout=0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class ASA(nn.Module):
    def __init__(self, dim, num_patches, heads=8, dim_head=64, dropout=0., is_LSA=False):
        super().__init__()
        inner_dim = dim_head * heads

        self.num_patches = num_patches
        self.heads = heads
        self.dim = dim
        self.scale = dim_head ** -0.5
        self.inner_dim = inner_dim
        self.is_BA = is_LSA

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # Project from inner_dim to dim
            nn.Dropout(dropout)
        ) if inner_dim != dim else nn.Identity()

        # Learnable weights for each head
        self.head_weights = nn.Parameter(torch.ones(heads))

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads
        d = self.inner_dim // h

        # Step 1: Linear projection to get Q, K, V
        qkv = self.to_qkv(x) # Should be (b, n, 3 * inner_dim)

        # Step 2: Split Q, K, V
        q, k, v = qkv.chunk(3, dim=-1) # Should be (b, n, inner_dim)

        # Step 3: Reshape and rearrange for multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q, k, v]) # Should be (b, h, n, d)

        # Step 4: Normalize Q and K
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1) # Should be (b, h, n, d)

        # Step 5: Compute attention scores (dots)
        dots = torch.matmul(q, k.transpose(-1, -2)) # Should be (b, h, n, n)

        # Step 6: Remove self-association
        mask = torch.eye(n, device=dots.device).bool()
        dots.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf')) # Should be (b, h, n, n)

        # Step 7: Optional BA mechanism
        if self.is_BA:
            q1 = torch.quantile(dots, 0.25, dim=-1)
            q3 = torch.quantile(dots, 0.75, dim=-1)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            topk = torch.sum(dots > upper_bound.unsqueeze(-1), dim=-1)
            topk_values, topk_indices = torch.topk(dots, topk.max(), dim=-1)
            mask = torch.full_like(dots, float('-inf'))
            dots = mask.scatter_(-1, topk_indices, topk_values) # Should be (b, h, n, n)

        # Step 8: Softmax to get attention probabilities
        attn = self.attend(dots) # Should be (b, h, n, n)

        # Step 9: Apply attention to V
        out = torch.matmul(attn, v) # Should be (b, h, n, d)

        # Step 10: Apply the weights to each head's output
        weighted_out = out * self.head_weights.view(1, h, 1, 1) # Should be (b, h, n, d)

        # Step 11: Concatenate the weighted outputs from all heads
        out = rearrange(weighted_out, 'b h n d -> b n (h d)') # Should be (b, n, inner_dim)
        #out = rearrange(out, 'b h n d -> b n (h d)') # Should be (b, n, inner_dim)

        # Step 12: Final projection
        out = self.to_out(out) # Should be (b, n, dim)

        return out

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


class DMSSCE3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., kernel_size=1,
                 with_bn=True, dim_head=0):
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
            nn.Conv3d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                      groups=hidden_features)
            for dilation in [1, 2, 4]  # Example dilation rates: 1, 2, 4
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
        cls_token, tokens = torch.split(x, [1, N - 1], dim=1)
        x = tokens.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        shortcut = x

        dilated_features = []
        for conv, bn in zip(self.dilated_convs, self.bn):
            dilated_features.append(self.act(bn(conv(x))))
        x = sum(dilated_features) / len(dilated_features)

        x = shortcut + x

        # Apply spatial SE block
        spatial_attention = self.sSE(x)
        x = x * spatial_attention

        # Channel SE Block
        x = self.cSE(x)

        x = self.conv3(x)
        x = self.bn3(x)

        tokens = x.flatten(2).permute(0, 2, 1)
        out = torch.cat((cls_token, tokens), dim=1)

        return out


class Dualmodules(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dim_head=64, act_layer=nn.GELU, drop=0.,
                 kernel_size=3, with_bn=True):
        super().__init__()

        self.scale = dim_head ** -0.5
        # Default settings
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 2D convolution layers for slice processing
        #self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)  # Pointwise convolution
        self.conv1 = WTConv2d(in_features, hidden_features, kernel_size=1, wt_levels=3)
        #self.conv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=hidden_features)  # Depthwise convolution
        self.conv2 = WTConv2d(hidden_features, hidden_features, kernel_size=kernel_size, wt_levels=3)
        #self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1)  # Pointwise convolution
        self.conv3 = WTConv2d(hidden_features, out_features, kernel_size=1, wt_levels=3)

        self.act = act_layer()

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.bn2 = nn.BatchNorm2d(hidden_features)
        self.bn3 = nn.BatchNorm2d(out_features)

        # Squeeze-and-Excitation (SE) mechanism
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Squeeze spatial dimensions (2D pooling)
        self.compress = nn.Linear(in_features, in_features // 4)
        self.excitation = nn.Linear(in_features // 4, in_features)
        self.SASFA = SASFA(dim=in_features, head_dim=dim_head)
        self.VMWTC = VMWTC(in_features=in_features, hidden_features=in_features, dim_head=dim_head, drop=drop)

    def forward(self, x, D=5, H=8, W=8):
        B, N, C = x.size()  # B: batch size, C: channels, D: depth, H: height, W: width
        wtsms  = self.SASFA(x)
        out = self.VMWTC(x)
        out =  out + wtsms  

        return out


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, DMSSCE, dropout=0., stochastic_depth=0.,
                 is_LSA=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}
        self.initials = DMSSCE
        self.depth = depth
        self.is_LSA = is_LSA
        

        for i in range(self.initials):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim, Dualmodules(in_features=dim, hidden_features=dim, out_features=dim,dim_head=dim_head)),
            ]))
        
        if is_LSA:
            for i in range(self.depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(num_patches, dim, ASA(dim=dim, num_patches=num_patches, heads=heads, dim_head=dim_head, dropout=dropout, is_LSA=self.is_LSA))
                ]))
        else:
            for i in range(self.depth - self.initials):
                self.layers.append(nn.ModuleList([
                    PreNorm(num_patches, dim, ASA(dim=dim, num_patches=num_patches, heads=heads, dim_head=dim_head, dropout=dropout, is_LSA=True)),
                    PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout=dropout))
                ]))
        

        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

    def forward(self, x):
        skip_connection = x.clone()
        for i, layer in enumerate(self.layers):
            attn = layer[0]
            x = self.drop_path(attn(x)) + x
            if len(layer) > 1:  # Check if FeedForward exists in the layer
                ff = layer[1]
                x = self.drop_path(ff(x)) + x
            self.scale[str(i)] = attn.fn.scale
            skip_connection = skip_connection.mean(dim=1, keepdim=True)
            x = x  + skip_connection
            skip_connection = x.clone()
            
        return x

class ViT3D(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio,DMSSCE, channels=1,
                 dim_head=16, dropout=0., emb_dropout=0., stochastic_depth=0., is_LSA=False, is_SPT=False):
        super().__init__()
        image_depth, image_height, image_width = 80,img_size,img_size
        patch_depth, patch_height, patch_width = patch_size,patch_size,patch_size
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)* (image_depth// patch_depth)
        self.patch_dim = channels * patch_height * patch_width * patch_depth
        self.dim = dim
        self.num_classes = num_classes
        self.fusion =EnhancedGatedMoE(input_dim=dim,num_experts=2)

        if not is_SPT:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', p1=patch_depth, p2=patch_height, p3=patch_width),
                nn.Linear(self.patch_dim, self.dim)
            )

        else:
            self.to_patch_embedding = est(in_channels=channels,high_res_channels=dim, embed_dim=dim)


        #self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))

        #self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer1 = Transformer(self.dim, self.num_patches, depth, heads, dim_head, mlp_dim_ratio,DMSSCE, dropout,
                                       stochastic_depth, is_LSA=True)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )

        self.apply(init_weights)

    def forward(self, img):
        x1 = self.to_patch_embedding(img)
        x1 = self.transformer1(x1)
        return self.mlp_head(x1[:, 0])
    

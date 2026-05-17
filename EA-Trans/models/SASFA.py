import torch
import torch.nn as nn
from typing import List

from models.WTConv import WTConv2d

class SASFA(nn.Module):
    def __init__(
            self, dim: int, head_dim: int, group_kernel_sizes: List[int] = [3, 5, 7, 9], 
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            gate_layer: str = 'sigmoid',
            wt_levels_sizes: List[int] = [3, 3, 3, 3]
    ):
        super(SASFA, self).__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        self.group_chans = group_chans = self.dim // 4

        # wt convolutions for each axis (depth, height, width)

        self.local_dwc_d = WTConv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[0], wt_levels=wt_levels_sizes[0])

        self.global_dwc_s_d = WTConv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[1], wt_levels=wt_levels_sizes[1])

        self.global_dwc_m_d = WTConv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[2], wt_levels=wt_levels_sizes[2])

        self.global_dwc_l_d = WTConv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[3], wt_levels=wt_levels_sizes[3])

        self.local_dwc_h = WTConv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[0], wt_levels=wt_levels_sizes[0])


        self.global_dwc_s_h = WTConv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[1], wt_levels=wt_levels_sizes[1])


        self.global_dwc_m_h = WTConv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[2], wt_levels=wt_levels_sizes[2])

        self.global_dwc_l_h = WTConv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[3], wt_levels=wt_levels_sizes[3])

        self.local_dwc_w = WTConv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[0], wt_levels=wt_levels_sizes[0])

        self.global_dwc_s_w = WTConv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[1], wt_levels=wt_levels_sizes[1])

        self.global_dwc_m_w = WTConv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[2], wt_levels=wt_levels_sizes[2])

        self.global_dwc_l_w = WTConv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[3], wt_levels=wt_levels_sizes[3])

        # gates and normalization layers
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_d = nn.GroupNorm(4, dim)
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.size()
        d = 5
        h = 8
        w = 8
        #cls_token, tokens = torch.split(x, [1, N - 1], dim=1)
        #x = tokens.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        x = x.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        #b, c, d, h, w = x.size()
        
        # Compute mean along each axis
        x_d = x.mean(dim=4)  # Averaged across width, shape: (b, c, d, h)
        x_h = x.mean(dim=3)  # Averaged across height, shape: (b, c, d, w)
        x_w = x.mean(dim=2)  # Averaged across depth, shape: (b, c, h, w)

        # Split channels
        l_x_d, g_x_d_s, g_x_d_m, g_x_d_l = torch.split(x_d, self.group_chans, dim=1)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        # Depth 
        x_d_attn = self.sa_gate(self.norm_d(torch.cat((
            self.local_dwc_d(l_x_d),
            self.global_dwc_s_d(g_x_d_s),
            self.global_dwc_m_d(g_x_d_m),
            self.global_dwc_l_d(g_x_d_l),
        ), dim=1)))  # Shape: (b, c, d, h)
        x_d_attn = x_d_attn.unsqueeze(-1)  # Shape: (b, c, d, h, 1)

        # Height 
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc_h(l_x_h),
            self.global_dwc_s_h(g_x_h_s),
            self.global_dwc_m_h(g_x_h_m),
            self.global_dwc_l_h(g_x_h_l),
        ), dim=1)))  # Shape: (b, c, d, w)
        x_h_attn = x_h_attn.unsqueeze(-2)  # Shape: (b, c, d, 1, w)

        # Width 
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc_w(l_x_w),
            self.global_dwc_s_w(g_x_w_s),
            self.global_dwc_m_w(g_x_w_m),
            self.global_dwc_l_w(g_x_w_l),
        ), dim=1)))  # Shape: (b, c, h, w)
        x_w_attn = x_w_attn.unsqueeze(2)  # Shape: (b, c, 1, h, w)

        # Element-wise multiplication across all attention maps
        x = x * x_d_attn * x_h_attn * x_w_attn


        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(b, d * h * w, c)
        
        return x


# Define input tensor shape
"""batch_size = 32
seq_length = 320  # Number of tokens
embedding_dim = 192  # Dimensionality of each token's embedding

# Initialize the SCSA3D module
SASFA = SASFA(dim=embedding_dim, head_dim=48)

# Generate a random input tensor with the specified shape
input_tensor = torch.randn(batch_size, seq_length, embedding_dim)

# Perform a forward pass through the model
output_tensor = SASFA(input_tensor)

# Print the shape of the output tensor
print(f"Output shape: {output_tensor.shape}")
"""
import pywt
import pywt.data
import torch
import torch.nn.functional as F
import torch.nn as nn

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False
def create_wavelet_filter(wave, in_size, out_size, type=torch.float, device='cpu'):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type, device=device)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type, device=device)
    dec_filters = torch.stack([dec_lo.view(1, 1, -1) * dec_lo.view(1, -1, 1) * dec_lo.view(-1, 1, 1),
                               dec_lo.view(1, 1, -1) * dec_lo.view(1, -1, 1) * dec_hi.view(-1, 1, 1),
                               dec_lo.view(1, 1, -1) * dec_hi.view(1, -1, 1) * dec_lo.view(-1, 1, 1),
                               dec_lo.view(1, 1, -1) * dec_hi.view(1, -1, 1) * dec_hi.view(-1, 1, 1),
                               dec_hi.view(1, 1, -1) * dec_lo.view(1, -1, 1) * dec_lo.view(-1, 1, 1),
                               dec_hi.view(1, 1, -1) * dec_lo.view(1, -1, 1) * dec_hi.view(-1, 1, 1),
                               dec_hi.view(1, 1, -1) * dec_hi.view(1, -1, 1) * dec_lo.view(-1, 1, 1),
                               dec_hi.view(1, 1, -1) * dec_hi.view(1, -1, 1) * dec_hi.view(-1, 1, 1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type, device=device).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type, device=device).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.view(1, 1, -1) * rec_lo.view(1, -1, 1) * rec_lo.view(-1, 1, 1),
                               rec_lo.view(1, 1, -1) * rec_lo.view(1, -1, 1) * rec_hi.view(-1, 1, 1),
                               rec_lo.view(1, 1, -1) * rec_hi.view(1, -1, 1) * rec_lo.view(-1, 1, 1),
                               rec_lo.view(1, 1, -1) * rec_hi.view(1, -1, 1) * rec_hi.view(-1, 1, 1),
                               rec_hi.view(1, 1, -1) * rec_lo.view(1, -1, 1) * rec_lo.view(-1, 1, 1),
                               rec_hi.view(1, 1, -1) * rec_lo.view(1, -1, 1) * rec_hi.view(-1, 1, 1),
                               rec_hi.view(1, 1, -1) * rec_hi.view(1, -1, 1) * rec_lo.view(-1, 1, 1),
                               rec_hi.view(1, 1, -1) * rec_hi.view(1, -1, 1) * rec_hi.view(-1, 1, 1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, d, h, w = x.shape
    pad = (filters.shape[2] // 2, filters.shape[3] // 2, filters.shape[4] // 2)
    x = F.conv3d(x, filters, stride=(2, 2, 2), groups=c, padding=pad)
    d_out = (d + pad[0] * 2 - filters.shape[2]) // 2 + 1
    h_out = (h + pad[1] * 2 - filters.shape[3]) // 2 + 1
    w_out = (w + pad[2] * 2 - filters.shape[4]) // 2 + 1
    x = x.reshape(b, c, 8, d_out, h_out, w_out)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, d_half, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2, filters.shape[3] // 2, filters.shape[4] // 2)
    x = x.reshape(b, c * 8, d_half, h_half, w_half)
    x = F.conv_transpose3d(x, filters, stride=(2, 2, 2), groups=c, padding=pad)
    return x

class WTConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv3d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv3d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv3d(in_channels*8, in_channels*8, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*8, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*8,1,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool3d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):
        device = x.device
        self.wt_filter = self.wt_filter.to(device)
        self.iwt_filter = self.iwt_filter.to(device)

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0) or (curr_shape[4] % 2 > 0):
                curr_pads = (0, curr_shape[4] % 2, 0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:,:,0,:,:,:]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 8, shape_x[3], shape_x[4], shape_x[5])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:8,:,:,:])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3], :curr_shape[4]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight.to(x.device), x)
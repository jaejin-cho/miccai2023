##########################################################
# %%
# import libraries
##########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##########################################################
# %%
# functions
##########################################################

def c2r(x):
    return torch.stack([x.real, x.imag], dim=-1)

def r2c(x):
    return torch.complex(x[..., 0], x[..., 1])

def tconj(x):
    return c2r(torch.conj(r2c(x)))


class tfft2(nn.Module):
    def __init__(self):
        super(tfft2, self).__init__()

    def forward(self, x):
        xc = r2c(x)
        # fft2 over the last two dimensions
        xt = torch.fft.fftshift(xc, dim=(-2, -1))
        kt = torch.fft.fft2(xt)
        kt = torch.fft.fftshift(kt, dim=(-2, -1))
        return c2r(kt)

class tifft2(nn.Module):
    def __init__(self):
        super(tifft2, self).__init__()

    def forward(self, x):
        xc = r2c(x)
        # ifft2 over the last two dimensions
        it = torch.fft.ifftshift(xc, dim=(-2, -1))
        it = torch.fft.ifft2(it)
        it = torch.fft.ifftshift(it, dim=(-2, -1))
        return c2r(it)


class rm_bg(nn.Module):
    def __init__(self):
        super(rm_bg, self).__init__()

    def forward(self, x):
        img, csm = x
        rcsm = torch.sum(torch.abs(csm), dim=1, keepdim=True)
        cmask = (rcsm > 0).float()[:,0,].unsqueeze(-1)
        rec = cmask * img
        return rec

              
class RegConvLayers(nn.Module):
    def __init__(self, nx, ny, ne, nLayers, num_filters):
        super(RegConvLayers, self).__init__()
        self.layers = nn.ModuleList()
        filter_size = (3, 3)

        self.layers.append(nn.Conv2d(2*ne, num_filters, filter_size, padding='same', bias=True))
        self.layers.append(nn.BatchNorm2d(num_filters))
        self.layers.append(nn.LeakyReLU(negative_slope=0.3))

        for _ in range(nLayers - 1):
            self.layers.append(nn.Conv2d(num_filters, num_filters, filter_size, padding='same', bias=True))
            self.layers.append(nn.BatchNorm2d(num_filters))
            self.layers.append(nn.LeakyReLU(negative_slope=0.3))

        self.final_layer = nn.Conv2d(num_filters, 2*ne, kernel_size=(1, 1), padding='same', bias=False)

    def forward(self, x):
        skip = x
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = x + skip
        return x



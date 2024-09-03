##########################################################
# %%
# import libraries
##########################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from library_net_function  import *
from library_net_recon     import create_recon

##########################################################
# %%
# forward model
##########################################################

class pForward(nn.Module):
    def __init__(self):
        super(pForward, self).__init__()

    def forward(self, x):
        o1, c1, m1 = x

        def forward1(tmp):
            o2, c2, m2 = tmp
            CI = c2 * o2.unsqueeze(0)
            kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(CI, dim=(-2, -1))), dim=(-2, -1))
            masked = kspace * m2.unsqueeze(0)
            return masked

        inp1 = (o1[..., 0], c1, m1)
        rec = torch.stack([forward1(t) for t in zip(*inp1)])
        return rec

##########################################################
# %%
# network
##########################################################
    
class ZeroMIRIDModel(nn.Module):
    def __init__(self, nx, ny, nLayers, num_block, num_filters=64):
        super(ZeroMIRIDModel, self).__init__()
        self.recon_joint    =   create_recon(nx=nx, ny=ny, num_block=num_block, nLayers=nLayers, num_filters=num_filters)
        self.oForward       =   pForward()

    def forward(self, inputs):
        input_c, input_k_trn1, input_k_trn2, input_k_lss1, input_k_lss2, input_m_trn1, input_m_trn2, input_m_lss1, input_m_lss2 = inputs

        out1, out2 = self.recon_joint([input_c, input_m_trn1, input_m_trn2, input_k_trn1, input_k_trn2])
        out3 = r2c(out1).unsqueeze(-1)
        out4 = r2c(out2).unsqueeze(-1)

        lss_1st = self.oForward([out3, input_c, input_m_lss1 + input_m_trn1])
        lss_2nd = self.oForward([out4, input_c, input_m_lss2 + input_m_trn2])

        lss_l1 = torch.sum(torch.abs(lss_1st - input_k_lss1 - input_k_trn1), dim=1) / (
            torch.sum(torch.abs(input_k_lss1 + input_k_trn1)) + torch.sum(torch.abs(input_k_lss2 + input_k_trn2))
        ) + torch.sum(torch.abs(lss_2nd - input_k_lss2 - input_k_trn2), dim=1) / (
            torch.sum(torch.abs(input_k_lss1 + input_k_trn1)) + torch.sum(torch.abs(input_k_lss2 + input_k_trn2))
        )

        lss_l2 = torch.sum(torch.square(torch.abs(lss_1st - input_k_lss1 - input_k_trn1)), dim=1) / torch.sqrt(
            torch.sum(torch.square(torch.abs(input_k_lss1 + input_k_trn1)) + torch.square(torch.abs(input_k_lss2 + input_k_trn2)))
        ) + torch.sum(torch.square(torch.abs(lss_2nd - input_k_lss2 - input_k_trn2)), dim=1) / torch.sqrt(
            torch.sum(torch.square(torch.abs(input_k_lss1 + input_k_trn1)) + torch.square(torch.abs(input_k_lss2 + input_k_trn2)))
        )

        out_final = torch.cat([
            out3, out4,
            lss_l1.unsqueeze(-1).to(torch.complex64),
            lss_l2.unsqueeze(-1).to(torch.complex64)
        ], dim=-1)

        return out_final



##########################################################
# %%
# custom loss
##########################################################

def loss_custom_v1(y_pred):
    # L1 norm
    l1 = torch.sum(torch.abs(y_pred[..., -2]))
    # L2 norm
    l2 = torch.sqrt(torch.sum(torch.abs(y_pred[..., -1])))

    return l1 + l2



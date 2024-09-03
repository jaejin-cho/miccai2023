##########################################################
# %%
# import libraries
##########################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import dft
from library_net_function import *


##########################################################
# %%
# SENSE
##########################################################

class Aclass:
    def __init__(self, csm, mask, lam):
        self.mask = r2c(mask)
        self.csm = r2c(csm)
        self.lam = lam
        self.nx, self.ny = csm.shape[1:3]

    def myAtA(self, img):
        coilImages = self.csm * img.unsqueeze(0)
        kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(coilImages, dim=(-2, -1))), dim=(-2, -1))
        temp = kspace * self.mask.unsqueeze(0)
        coilImgs = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(temp, dim=(-2, -1))), dim=(-2, -1))
        coilComb = torch.sum(coilImgs * torch.conj(self.csm), dim=0)
        coilComb = coilComb + self.lam * img
        return coilComb



def myCG(A, rhs):
    rhs = r2c(rhs)
    x = torch.zeros_like(rhs)
    r = rhs
    p = rhs
    rTr = torch.real(torch.sum(torch.conj(r) * r))
    i = 0

    while i < 10 and rTr > 1e-8:
        Ap = A.myAtA(p)
        alpha = rTr / torch.real(torch.sum(torch.conj(p) * Ap))
        alpha = torch.complex(alpha, torch.tensor(0.0).cuda())
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.real(torch.sum(torch.conj(r) * r))
        beta = rTrNew / rTr
        beta = torch.complex(beta, torch.tensor(0.0).cuda())
        p = r + beta * p
        rTr = rTrNew
        i += 1

    return c2r(x)



class myDC(nn.Module):
    def __init__(self):
        super(myDC, self).__init__()
        self.lam1 = nn.Parameter(torch.tensor(0.03, dtype=torch.float32)).cuda()
        self.lam2 = nn.Parameter(torch.tensor(0.03, dtype=torch.float32)).cuda()

    def forward(self, x):
        rhs, csm, mask = x
        lam3 = torch.complex(self.lam1 + self.lam2, torch.tensor(0.0).cuda())

        def fn(tmp):
            c, m, r = tmp
            Aobj = Aclass(c, m, lam3)
            y = myCG(Aobj, r)
            return y

        inp = (csm, mask, rhs)
        rec = torch.stack([fn(x) for x in zip(*inp)])
        return rec

    def lam_weight(self, x):
        in0, in1 = x
        res = self.lam1 * in0 + self.lam2 * in1
        return res



class Aty(nn.Module):
    def __init__(self):
        super(Aty, self).__init__()

    def forward(self, x):
        kdata, csm, mask = x

        def backward(tmp):
            k, c, m = tmp
            ks = k * m.unsqueeze(0)
            ci = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(ks, dim=(-2, -1))), dim=(-2, -1))
            y1 = torch.sum(ci * torch.conj(c), dim=0)
            return y1

        inp = (kdata, csm, mask)
        rec = torch.stack([backward(x) for x in zip(*inp)])
        return rec


class create_recon(nn.Module):
    def __init__(self, nx, ny, nLayers, num_block, num_filters=64):
        super(create_recon, self).__init__()
        self.nLayers = nLayers
        self.num_block = num_block
        self.num_filters = num_filters
        self.RegConv_k = RegConvLayers(nx, ny, 4, nLayers, num_filters)
        self.RegConv_i = RegConvLayers(nx, ny, 4, nLayers, num_filters)
        self.calc_Aty  = Aty()
        self.UpdateDC = myDC()
        self.rmbg = rm_bg()
        self.myFFT = tfft2()
        self.myIFFT = tifft2()

    def forward(self, inputs):

        input_c, input_m1, input_m2, input_k1, input_k2 = inputs
        Atb1 = self.calc_Aty([input_k1, input_c, input_m1])
        Atb2 = self.calc_Aty([input_k2, input_c, input_m2])
        dc1 = c2r(Atb1)
        dc2 = c2r(Atb2)

        for blk in range(self.num_block):
            dc_cat_i = torch.cat([dc1, dc2, tconj(dc1), tconj(dc2)], dim=-1)
            dc_cat_k = torch.cat([self.myFFT(dc1), self.myFFT(dc2), self.myFFT(torch.conj(dc1)), self.myFFT(torch.conj(dc2))], dim=-1)

            rg_term_i = self.RegConv_i(dc_cat_i.permute((0,3,1,2))).permute((0,2,3,1))
            rg_term_k = self.RegConv_k(dc_cat_k.permute((0,3,1,2))).permute((0,2,3,1))

            irg1 = (rg_term_i[..., :2] + tconj(rg_term_i[..., 4:6])) / 2
            irg2 = (rg_term_i[..., 2:4] + tconj(rg_term_i[..., 6:8])) / 2
            krg1 = (self.myIFFT(rg_term_k[..., :2]) + tconj(self.myIFFT(rg_term_k[..., 4:6]))) / 2
            krg2 = (self.myIFFT(rg_term_k[..., 2:4]) + tconj(self.myIFFT(rg_term_k[..., 6:8]))) / 2
            rg1 = self.UpdateDC.lam_weight([irg1, krg1])
            rg2 = self.UpdateDC.lam_weight([irg2, krg2])

            rg1 = rg1 + c2r(Atb1)
            rg2 = rg2 + c2r(Atb2)

            dc1 = self.UpdateDC([rg1, c2r(input_c), c2r(input_m1)])
            dc2 = self.UpdateDC([rg2, c2r(input_c), c2r(input_m2)])

        out1 = self.rmbg([dc1, input_c])
        out2 = self.rmbg([dc2, input_c])

        return out1, out2


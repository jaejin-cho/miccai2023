##########################################################
# %%
# Library for tensorflow 2.2.0
##########################################################

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import h5py

##########################################################
# %%
# define common functions
##########################################################

def mosaic(img, num_row, num_col, fig_num, clim, title='', use_transpose=False, use_flipud=False):
    fig = plt.figure(fig_num)
    fig.patch.set_facecolor('black')

    if img.ndim < 3:
        img_res = img
        plt.imshow(img_res)
        plt.gray()
        plt.clim(clim)
    else:
        if img.shape[2] != (num_row * num_col):
            print('sizes do not match')
        else:
            if use_transpose:
                for slc in range(0, img.shape[2]):
                    img[:, :, slc] = np.transpose(img[:, :, slc])

            if use_flipud:
                img = np.flipud(img)

            img_res = np.zeros((img.shape[0] * num_row, img.shape[1] * num_col))
            idx = 0

            for r in range(0, num_row):
                for c in range(0, num_col):
                    img_res[r * img.shape[0]: (r + 1) * img.shape[0], c * img.shape[1]: (c + 1) * img.shape[1]] = img[:,
                                                                                                                  :,
                                                                                                                  idx]
                    idx = idx + 1
        plt.imshow(img_res)
        plt.gray()
        plt.clim(clim)

    plt.suptitle(title, color='white', fontsize=48)


def msave_img(filename,data,intensity):
     
    data    =   (data - intensity[0]) * 255 / (intensity[1]-intensity[0])
    data[data>255]  =   255
    data[data<0]    =   0
    img     =   Image.fromarray(data.astype(np.uint8))
    img.save(filename)
    
    return True

def mvec(data):
    xl = data.size
    res = np.reshape(data, (xl))
    return res


def load_h5py(filename, rmod='r'):
    f = h5py.File(filename, rmod)
    arr = {}
    for k, v in f.items():
        arr[k] = np.transpose(np.array(v))
    return arr


def mfft(x, axis=0):
    # nx = x.shape[axis]
    y = np.fft.fftshift(np.fft.fft(np.fft.fftshift(x, axes=axis), axis=axis), axes=axis) # / np.sqrt(nx)
    return y


def mifft(x, axis=0):
    # nx = x.shape[axis]
    y = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis) # * np.sqrt(nx)
    return y


def mfft2(x, axes=(0, 1)):
    # nx, ny = x.shape[axes]
    y = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x, axes=axes), axes=axes), axes=axes) # / np.sqrt(nx,ny)
    return y


def mifft2(x, axes=(0, 1)):
    # nx, ny = x.shape[axes]
    y = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes) # * np.sqrt(nx,ny)
    return y


def msos(img, axis=3):
    return np.sqrt(np.sum(np.abs(img) ** 2, axis=axis))


def uniform_selection(input_data, input_mask, rho=0.2, small_acs_block=(4, 4)):

    nrow, ncol = input_data.shape[0], input_data.shape[1]

    center_kx = int(find_center_ind(input_data, axes=(1, 2)))
    center_ky = int(find_center_ind(input_data, axes=(0, 2)))

    temp_mask = np.copy(input_mask)
    temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
    center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

    pr = np.ndarray.flatten(temp_mask)
    ind = np.random.choice(np.arange(nrow * ncol),
                            size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

    [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

    loss_mask = np.zeros_like(input_mask)
    loss_mask[ind_x, ind_y] = 1

    trn_mask = input_mask - loss_mask

    return trn_mask, loss_mask


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims: return tensor.squeeze()

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    center_locs = norm(kspace, axes=axes).squeeze()
    return np.argsort(center_locs)[-1:]


def index_flatten2nd(ind, shape):
    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]


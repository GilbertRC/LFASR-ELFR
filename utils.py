import torch
import torch.nn as nn
import torch.nn.functional as functional
import argparse
import numpy as np
import math
import copy


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr


class L1_Charbonnier_loss(torch.nn.Module):
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, gt):
        diff = torch.add(pred, -gt)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


def gradient(im):
    D_dy = im[:, :, 1:, :] - im[:, :, :-1, :]
    D_dx = im[:, :, :, 1:] - im[:, :, :, :-1]

    return D_dx, D_dy


def lf2epi(lf):
    batch_size, an, an, h, w = lf.shape
    # [b,an,an,h,w] -> [b*ah*h,aw,w] & [b*aw*w,ah,h]
    epi_h = lf.view(batch_size, an, an, h, w).permute(0, 1, 3, 2, 4).contiguous().view(-1, 1, an, w)
    epi_v = lf.view(batch_size, an, an, h, w).permute(0, 2, 4, 1, 3).contiguous().view(-1, 1, an, h)

    return epi_h, epi_v


class EPI_loss(torch.nn.Module):
    def __init__(self):
        super(EPI_loss, self).__init__()
        self.L1_loss = L1_Charbonnier_loss()

    def forward(self, pred, gt):
        epi_h_pred, epi_v_pred = lf2epi(pred)
        dx_h_pred, dy_h_pred = gradient(epi_h_pred)
        dx_v_pred, dy_v_pred = gradient(epi_v_pred)

        epi_h_gt, epi_v_gt = lf2epi(gt)
        dx_h_gt, dy_h_gt = gradient(epi_h_gt)
        dx_v_gt, dy_v_gt = gradient(epi_v_gt)

        loss = self.L1_loss(dx_h_pred, dx_h_gt) + self.L1_loss(dy_h_pred, dy_h_gt) \
               + self.L1_loss(dx_v_pred, dx_v_gt) + self.L1_loss(dy_v_pred, dy_v_gt)

        return loss


class smooth_loss(torch.nn.Module):
    def __init__(self, weight):
        super(smooth_loss, self).__init__()
        self.weight = weight

    def forward(self, pred_map):
        dx, dy = gradient(pred_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss = (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * self.weight

        return loss


def crop_boundary(I, crop_size):
    """crop the boundary (the last 2 dimensions) of a tensor"""
    if crop_size[0] == 0 and crop_size[1] == 0:
        return I

    if crop_size[0] > 0 or crop_size[1] > 0:
        size = list(I.shape)
        I_crop = I.view(-1, size[-2], size[-1])
        crop_y = int(crop_size[0])
        crop_x = int(crop_size[1])
        I_crop = I_crop[:, crop_y:-crop_y, crop_x:-crop_x]
        size[-1] -= crop_x * 2
        size[-2] -= crop_y * 2
        I_crop = I_crop.view(size)
        return I_crop


def warping(disp, ind_source, ind_target, img_source, an):
    """warping one source image/map to the target"""

    # disp:       [scale] or [N,h,w]
    # ind_souce:  (int)
    # ind_target: (int)
    # img_source: [N,h,w]
    # an:         angular number
    # ==> out:    [N,1,h,w]

    N, h, w = img_source.shape
    ind_source = ind_source.type_as(disp)
    ind_target = ind_target.type_as(disp)

    # coordinate for source and target
    ind_h_source = torch.floor(ind_source / an)
    ind_w_source = ind_source % an

    ind_h_target = torch.floor(ind_target / an)
    ind_w_target = ind_target % an

    # generate grid
    XX = torch.arange(0, w).view(1, 1, w).expand(N, h, w).type_as(img_source)  # [N,h,w]
    YY = torch.arange(0, h).view(1, h, 1).expand(N, h, w).type_as(img_source)

    grid_w = XX + disp * (ind_w_target - ind_w_source)
    grid_h = YY + disp * (ind_h_target - ind_h_source)

    grid_w_norm = 2.0 * grid_w / (w - 1) - 1.0  # 因为grid_sample需要归一化到[-1,1]
    grid_h_norm = 2.0 * grid_h / (h - 1) - 1.0

    grid = torch.stack((grid_w_norm, grid_h_norm), dim=3)  # [N,h,w,2]

    # inverse warp
    # img_source = torch.unsqueeze(img_source, 0)
    img_source = torch.unsqueeze(img_source, 1)
    img_target = functional.grid_sample(img_source, grid, align_corners=True)  # [N,1,h,w]
    img_target = torch.squeeze(img_target, 1)  # [N,h,w]

    return img_target


class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)


def ycbcr2rgb(ycbcr):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:, 0] -= 16. / 255.
    rgb[:, 1:] -= 128. / 255.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 1).reshape(shape).astype(np.float32)


def compt_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0

    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

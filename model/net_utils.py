import torch
import torch.nn as nn
import math
import numpy as np


def construct_psv_grid(an, num_target, idx_target, D, num_source, idx_src, disp_range, batch_size, h, w):
    grid = []
    for k_t in range(0, num_target):
        for step in range(0, D):
            for k_s in range(0, num_source):
                ind_s = idx_src[:, k_s].view(batch_size, 1, 1).type_as(disp_range)
                ind_t = idx_target[:, k_t].view(batch_size, 1, 1).type_as(disp_range)
                # ind_t = torch.arange(an * an)[k_t].type_as(disp_range)
                ind_s_h = torch.floor(ind_s / an)
                ind_s_w = ind_s % an
                ind_t_h = torch.floor(ind_t / an)
                ind_t_w = ind_t % an
                disp = disp_range[step]

                XX = torch.arange(0, w).view(1, 1, w).expand(batch_size, h, w).type_as(disp_range)  # [b,h,w]
                YY = torch.arange(0, h).view(1, h, 1).expand(batch_size, h, w).type_as(disp_range)

                grid_w_t = XX + disp * (ind_t_w - ind_s_w)
                grid_h_t = YY + disp * (ind_t_h - ind_s_h)

                grid_w_t_norm = 2.0 * grid_w_t / (w - 1) - 1.0
                grid_h_t_norm = 2.0 * grid_h_t / (h - 1) - 1.0

                grid_t = torch.stack((grid_w_t_norm, grid_h_t_norm), dim=3)  # [b,h,w,2]
                grid.append(grid_t)

    grid = torch.cat(grid, 0)  # [b*nT*D*4,h,w,2]
    return grid


def construct_syn_grid(an, num_target, idx_target, num_source, idx_src, disp_target, batch_size, h, w):
    grid = []
    for k_t in range(0, num_target):
        for k_s in range(0, num_source):
            ind_s = idx_src[:, k_s].view(batch_size, 1, 1).type_as(disp_target)
            ind_t = idx_target[:, k_t].view(batch_size, 1, 1).type_as(disp_target)
            # ind_t = torch.arange(an * an)[k_t].type_as(disp_target)
            ind_s_h = torch.floor(ind_s / an)
            ind_s_w = ind_s % an
            ind_t_h = torch.floor(ind_t / an)
            ind_t_w = ind_t % an
            disp = disp_target[:, k_t, :, :]

            XX = torch.arange(0, w).view(1, 1, w).expand(batch_size, h, w).type_as(disp_target)  # [b,h,w]
            YY = torch.arange(0, h).view(1, h, 1).expand(batch_size, h, w).type_as(disp_target)

            grid_w_t = XX + disp * (ind_t_w - ind_s_w)
            grid_h_t = YY + disp * (ind_t_h - ind_s_h)

            grid_w_t_norm = 2.0 * grid_w_t / (w - 1) - 1.0
            grid_h_t_norm = 2.0 * grid_h_t / (h - 1) - 1.0

            grid_t = torch.stack((grid_w_t_norm, grid_h_t_norm), dim=3)  # [b,h,w,2]
            grid.append(grid_t)

    grid = torch.cat(grid, 0)  # [b*nT*4,h,w,2]
    return grid


def replace_and_reshape(target_view, src_views, an, idx_src, idx_target):
    an2 = an * an
    batch_size, num_source, h, w = src_views.shape

    # index
    index = torch.zeros(batch_size, an2, dtype=torch.int64).to(src_views.device)
    index[:, idx_src.long()] = torch.from_numpy(np.arange(an2 - num_source, an2, 1, dtype=np.int64)).to(src_views.device)
    index[:, idx_target.long()] = torch.from_numpy(np.arange(an2 - num_source, dtype=np.int64)).to(src_views.device)

    output_view = torch.cat((target_view, src_views), 1)
    output_view = output_view[:, index]

    return output_view.view(batch_size, an, an, h, w)


def replace(target_view, src_views, an, idx_src, idx_target):
    an2 = an * an
    batch_size, num_source, h, w = src_views.shape

    # index
    index = torch.zeros(batch_size, an2, dtype=torch.int64).to(src_views.device)
    index[:, idx_src.long()] = torch.from_numpy(np.arange(an2 - num_source, an2, 1, dtype=np.int64)).to(src_views.device)
    index[:, idx_target.long()] = torch.from_numpy(np.arange(an2 - num_source, dtype=np.int64)).to(src_views.device)

    output_view = torch.cat((target_view, src_views), 1)
    output_view = output_view[:, index]

    return output_view.view(batch_size, an2, h, w)


class ResBlock3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding)
        if stride == (1, 1, 1):
            self.downsample = None
        elif stride == (2, 2, 2):
            self.downsample = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1), stride=stride)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)

        if self.downsample is not None:
            x = self.downsample(x)

        res += x

        return res


def make_ResBlock3d(n_blocks, n_feats, kernel_size, stride=(1, 1, 1), padding=(1, 1, 1)):
    layers = []
    for i in range(n_blocks):
        layers.append(ResBlock3d(n_feats, n_feats, kernel_size, stride, padding))
    return nn.Sequential(*layers)

import torch
import torch.nn as nn
import torch.nn.functional as functional
import math

from model.net_utils import construct_psv_grid, construct_syn_grid, make_ResBlock3d, replace


class Net(nn.Module):
    """explicit & implicit depth-based view synthesis network"""

    def __init__(self, opt):
        super(Net, self).__init__()
        self.an = opt.angular_out
        self.an2 = opt.angular_out * opt.angular_out

        self.feature_extraction = nn.Sequential(
            nn.Conv3d(1, self.an2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.an2, self.an2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.an2, self.an2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.feature_implicit = nn.Sequential(
            nn.Conv3d(opt.num_planes, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.feature_explicit = nn.Sequential(
            nn.Conv3d(self.an2, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.residual_learning = nn.Sequential(
            nn.Conv3d(self.an2, self.an2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            make_ResBlock3d(n_blocks=8, n_feats=self.an2, kernel_size=(3, 3, 3))
        )

        self.disp_estimation = nn.Sequential(
            nn.Conv2d(opt.num_planes * opt.num_source, self.an2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.an2, 32, kernel_size=(7, 7), stride=(1, 1), dilation=(2, 2), padding=(6, 6)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(7, 7), stride=(1, 1), dilation=(2, 2), padding=(6, 6)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.an2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.an2, self.an2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.an2, self.an2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.an2, self.an2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.an2, self.an2 - opt.num_source, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.view_synthesis = nn.Sequential(
            nn.Conv3d(self.an2, self.an2, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.an2, self.an2, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.an2, self.an2 - opt.num_source, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
        )

        self.refinement_final = nn.Sequential(
            nn.Conv3d(in_channels=opt.num_source, out_channels=self.an2, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            make_ResBlock3d(n_blocks=4, n_feats=self.an2, kernel_size=(3, 3, 3)),
            nn.Conv3d(in_channels=self.an2, out_channels=self.an2, kernel_size=(5, 3, 3), stride=(4, 1, 1), padding=(0, 1, 1)),  # 49-->12
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.an2, out_channels=self.an2, kernel_size=(4, 3, 3), stride=(4, 1, 1), padding=(0, 1, 1)),  # 12-->3
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.an2, out_channels=self.an2 - opt.num_source, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),  # 3-->1
        )

        self.disp_thres = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda())
        self.scale = torch.nn.Parameter(torch.tensor(20, dtype=torch.float32, requires_grad=True).cuda())
        self.sigmoid = nn.Sigmoid()

    def forward(self, idx_src, src_views, idx_target, opt):

        an = opt.angular_out
        an2 = opt.angular_out * opt.angular_out

        batch_size, num_source, h, w = src_views.shape  # [b,n,h,w]
        num_target = idx_target.shape[1]
        idx_center = idx_target[:, math.floor(num_target / 2)].unsqueeze(1)

        D = opt.num_planes
        disp_range = torch.linspace(-1 * opt.disp_range, opt.disp_range, steps=D).type_as(src_views)  # [D]

        # PSV
        psv_input = src_views.view(batch_size * num_source, 1, h, w).repeat(D, 1, 1, 1)  # [b*D*4,1,h,w]
        grid = construct_psv_grid(an, 1, idx_center, D, num_source, idx_src, disp_range, batch_size, h, w)  # [b*1*D*4,h,w,2]
        PSV = functional.grid_sample(psv_input, grid, align_corners=True).view(batch_size, 1, D, num_source, h, w)  # [b*1*D*4,1,h,w]-->[b,1,D,4,h,w]

        # feature extraction and decoupling
        feature_common = self.feature_extraction(PSV.view(batch_size * 1 * D, 1, num_source, h, w))  # [b*D,an2,4,h,w]
        feature_common = feature_common.view(batch_size, D, an2, num_source, h, w)  # [b,D,an2,4,h,w]
        feature_implicit = torch.transpose(feature_common, 1, 2).view(batch_size * an2, D, num_source, h, w)  # [b,an2,D,4,h,w]
        feature_implicit = self.feature_implicit(feature_implicit).view(batch_size, an2, num_source, h, w)  # [b,an2,4,h,w]
        feature_explicit = feature_common.view(batch_size * D, an2, num_source, h, w)
        feature_explicit = self.feature_explicit(feature_explicit).view(batch_size, D * num_source, h, w)  # [b,an2,h,w]

        # explicit depth-based pipeline
        disp_target = self.disp_estimation(feature_explicit)  # [b,nT,h,w]
        warp_img_input = src_views.view(batch_size * num_source, 1, h, w).repeat(num_target, 1, 1, 1)  # [b*nT*4,1,h,w]
        grid = construct_syn_grid(an, num_target, idx_target, num_source, idx_src, disp_target, batch_size, h,w)  # [b*nT*4,h,w,2]
        view_explicit = functional.grid_sample(warp_img_input, grid, align_corners=True).view(batch_size, num_target, num_source, h, w)  # [b,nT,4,h,w]

        # implicit depth-based pipeline
        feature_implicit = self.residual_learning(feature_implicit)
        view_implicit = self.view_synthesis(feature_implicit).view(batch_size, num_target, h, w)

        # disparity-guided combination
        # mask→implicit
        mask = self.scale * (torch.abs(disp_target) - self.disp_thres)
        mask = 1 - self.sigmoid(mask)
        view_ehc_0 = (1 - mask) * view_explicit[:, :, 0, :, :] + mask * view_implicit
        view_ehc_1 = (1 - mask) * view_explicit[:, :, 1, :, :] + mask * view_implicit
        view_ehc_2 = (1 - mask) * view_explicit[:, :, 2, :, :] + mask * view_implicit
        view_ehc_3 = (1 - mask) * view_explicit[:, :, 3, :, :] + mask * view_implicit

        # fusion
        ehc_0 = replace(view_ehc_0, src_views, an, idx_src, idx_target)
        ehc_1 = replace(view_ehc_1, src_views, an, idx_src, idx_target)
        ehc_2 = replace(view_ehc_2, src_views, an, idx_src, idx_target)
        ehc_3 = replace(view_ehc_3, src_views, an, idx_src, idx_target)
        res_group_final = torch.cat((ehc_0.unsqueeze(1), ehc_1.unsqueeze(1),
                                     ehc_2.unsqueeze(1), ehc_3.unsqueeze(1)), dim=1)
        res_fuse = self.refinement_final(res_group_final).view(batch_size, num_target, h, w)  # [N,nT,h,w]
        output_view = view_ehc_0 + res_fuse

        return disp_target, view_explicit, view_implicit, output_view, mask, self.disp_thres

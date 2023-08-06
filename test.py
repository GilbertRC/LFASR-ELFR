import torch
import torch.utils.data as data
import torch.nn.functional as functional
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
from os import listdir
from os.path import join

import math
from math import ceil, floor
import pandas as pd
from PIL import Image
import h5py
from scipy import misc
from skimage.measure import compare_ssim

import data_loader
import utils
from model.models import Net
from model.net_utils import replace

# ----------------------------------------------------------------------------------#
# Test settings
parser = argparse.ArgumentParser(description='PyTorch Light Field Reconstruction Testing')
parser.add_argument('--experiment_name', type=str, default='model_2x2-7x7(SIG)',
                    help='name for the experiment to run')
parser.add_argument('--resume_epoch', type=int, default='7000', help='epoch to resume')
parser.add_argument('--model_dir', type=str, default='pretrained_model', help='directory for model checkpoints')
parser.add_argument('--save_dir', type=str, default='results', help='folder to save the test results')
parser.add_argument('--angular_in', type=int, default=2,
                    help='angular number of the sparse light field, [AngIn x AngIn](fixed) or AngIn(random)')
parser.add_argument('--angular_out', type=int, default=7,
                    help='angular number of the dense light field [AngOut x AngOut]')
parser.add_argument('--disp_range', type=float, default=1.5, help='depth range for psv')
parser.add_argument('--num_planes', type=int, default=32, help='step number for psv')
parser.add_argument('--train_dataset', type=str, default='SIG', help='dataset for training')
parser.add_argument('--dataset_path', type=str, default='./LFData/', help='dataset path for testing')
parser.add_argument('--input_ind', action=utils.Store_as_array, type=int, nargs='+')
parser.add_argument('--save_img', type=int, default=1, help='save image or not')
parser.add_argument('--crop', type=int, default=0, help='crop the image into patches when out of memory')
parser.add_argument('--ids', type=str, default='3', help='set the cuda devices')
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.ids


def min_max_norm(in_):
    max_ = in_.max()
    min_ = in_.min()
    in_ = in_ - min_
    return in_ / (max_ - min_ + 1e-8)


def depth_norm(in_):
    max_ = opt.disp_range
    min_ = -opt.disp_range
    in_ = in_ - min_
    return in_ / (max_ - min_ + 1e-8)


def CropPatches_w(image, len, crop):
    # image [1,an2,ph,pw]
    # left [1,an2,h,lw]
    # middles[n,an2,h,mw]
    # right [1,an2,h,rw]
    batch_size, an, h, w = image.shape[0:4]
    left = image[:, :, :, 0:len + crop]
    num = math.floor((w - len - crop) / len)
    middles = torch.Tensor(batch_size, num, an, h, len + crop * 2).to(image.device)
    for i in range(num):
        middles[:, i] = image[:, :, :, (i + 1) * len - crop:(i + 2) * len + crop]
    right = image[:, :, :, (num + 1) * len - crop:]
    return left, middles, right


def MergePatches_w(left, middles, right, h, w, len, crop):
    # [N,4,h,w]
    n, a = left.shape[0:2]
    out = torch.Tensor(n, a, h, w).to(left.device)
    out[:, :, :, :len] = left[:, :, :, :-crop]
    for i in range(middles.shape[1]):
        out[:, :, :, len * (i + 1):len * (i + 2)] = middles[:, i, :, :, crop:-crop]
    out[:, :, :, len * (middles.shape[1] + 1):] = right[:, :, :, crop:]
    return out


def save_results(LF, lfi_ycbcr, target_y, lf_no, opt):
    # save image
    if opt.save_img:
        for i in range(opt.angular_out * opt.angular_out):
            img_ycbcr = lfi_ycbcr[0, i]  # using gt ycbcr for visual results
            img_ycbcr[:, :, 0] = LF[0, i]  # [h,w,3]
            img_name = '{}/SynLFI{}_view{}.png'.format(opt.save_img_dir, lf_no, i)
            img_rgb = utils.ycbcr2rgb(img_ycbcr)
            img = (img_rgb.clip(0, 1) * 255.0).astype(np.uint8)
            # misc.toimage(img, cmin=0, cmax=255).save(img_name)
            Image.fromarray(img).convert('RGB').save(img_name)

    # compute psnr/ssim
    view_list = []
    view_psnr_y = []
    view_ssim_y = []
    for i in range(opt.angular_out * opt.angular_out):
        if i not in opt.input_ind:
            cur_target_y = target_y[0, i]
            cur_y = LF[0, i]
            cur_psnr_y = utils.compt_psnr(cur_target_y, cur_y)
            cur_ssim_y = compare_ssim((cur_target_y * 255.0).astype(np.uint8),
                                      (cur_y * 255.0).astype(np.uint8),
                                      gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            view_list.append(i)
            view_psnr_y.append(cur_psnr_y)
            view_ssim_y.append(cur_ssim_y)

    dataframe_lfi = pd.DataFrame(
        {'targetView_LFI{}'.format(lf_no): view_list, 'psnr Y': view_psnr_y, 'ssim Y': view_ssim_y})
    dataframe_lfi.to_csv(opt.save_csv_name, index=False, sep=',', mode='a')

    return np.mean(view_psnr_y), np.mean(view_ssim_y)


def test(test_loader, model):
    # testing
    model.eval()

    lf_list = []
    lf_psnr_list = []
    lf_ssim_list = []
    with torch.no_grad():
        for k, batch in enumerate(test_loader):
            src_views, idx_all, target_y, lfi_ycbcr = batch[0], batch[1], batch[2].numpy(), batch[3].numpy()
            src_views = utils.crop_boundary(src_views.cuda(), opt.crop_size)
            idx_src = torch.from_numpy(opt.input_ind).unsqueeze(0)
            idx_target = idx_all[torch.from_numpy(np.isin(idx_all, idx_src, invert=True))].unsqueeze(0)

            if not opt.crop:
                disp, _, _, LF, _, _ = model(idx_src, src_views, idx_target, opt)
            else:
                length = 120
                crop = 20
                input_l, input_m, input_r = CropPatches_w(src_views, length, crop)
                disp_l, _, _, pred_l, _, _ = model(idx_src, input_l, idx_target, opt)
                pred_m = torch.Tensor(input_m.shape[0], input_m.shape[1], opt.angular_out * opt.angular_out - 4, input_m.shape[3], input_m.shape[4])
                disp_m = torch.Tensor(input_m.shape[0], input_m.shape[1], opt.angular_out * opt.angular_out - 4, input_m.shape[3], input_m.shape[4])
                for i in range(input_m.shape[1]):
                    disp_m[:, i], _, _, pred_m[:, i], _, _ = model(idx_src, input_m[:, i], idx_target, opt)
                disp_r, _, _, pred_r, _, _ = model(idx_src, input_r, idx_target, opt)
                LF = MergePatches_w(pred_l, pred_m, pred_r, src_views.shape[2], src_views.shape[3], length, crop)  # [N,an2,hs,ws]
                disp = MergePatches_w(disp_l, disp_m, disp_r, src_views.shape[2], src_views.shape[3], length, crop)  # [N,an2,hs,ws]

            LF = replace(LF, src_views, opt.angular_out, idx_src, idx_target)
            LF = utils.crop_boundary(functional.pad(LF, pad=[opt.crop_size[1], opt.crop_size[1], opt.crop_size[0], opt.crop_size[0]],
                                                    mode='constant', value=0), opt.test_crop_size)
            LF = LF.cpu().numpy()

            bd = opt.test_crop_size
            bd_y = int(bd[0])
            bd_x = int(bd[1])
            target_y = target_y[:, :, bd_y:-bd_y, bd_x:-bd_x]
            lfi_ycbcr = lfi_ycbcr[:, :, bd_y:-bd_y, bd_x:-bd_x, :]

            lf_psnr, lf_ssim = save_results(LF, lfi_ycbcr, target_y, k, opt)

            lf_list.append(k)
            lf_psnr_list.append(lf_psnr)
            lf_ssim_list.append(lf_ssim)

        dataframe_lfi = pd.DataFrame({'LFI': lf_list, 'psnr Y': lf_psnr_list, 'ssim Y': lf_ssim_list})
        dataframe_lfi.to_csv(opt.save_csv_name, index=False, sep=',', mode='a')

        dataframe_lfi = pd.DataFrame(
            {'summary': ['avg'], 'psnr Y': [np.mean(lf_psnr_list)], 'ssim Y': [np.mean(lf_ssim_list)]})
        dataframe_lfi.to_csv(opt.save_csv_name, index=False, sep=',', mode='a')


if __name__ == '__main__':
    # device configuration
    num_gpu = torch.cuda.device_count()
    device_ids = list(range(num_gpu))
    print('Using Gpu: {}'.format(opt.ids))

    # model params
    opt.num_source = opt.angular_in * opt.angular_in
    opt.crop_size = np.array([0, 0])  # crop_y, crop_x
    opt.test_crop_size = np.array([22, 22])

    # build model
    print("Building net")
    model = Net(opt).cuda()

    # load pretrained model
    resume_path = join(opt.model_dir, opt.experiment_name, 'lfr.pth.{}'.format(opt.resume_epoch))
    model.load_state_dict(torch.load(resume_path))
    print('Loaded model {}'.format(resume_path))

    test_datasets = []
    if opt.train_dataset == 'SIG':
        test_datasets = ['30scenes', 'occlusions', 'reflective']
    elif opt.train_dataset == 'HCI':
        test_datasets = ['HCI_old', 'HCI']

    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    for dataset in test_datasets:
        # generate save folder
        opt.save_img_dir = '{}/res_{}(epoch{})_input{}/{}'.format(opt.save_dir, opt.experiment_name,
                                                                  opt.resume_epoch, opt.input_ind,
                                                                  dataset)
        if not os.path.exists(opt.save_img_dir):
            os.makedirs(opt.save_img_dir)
        opt.save_csv_name = '{}/res_{}(epoch{})_input{}/{}.csv'.format(opt.save_dir, opt.experiment_name,
                                                                       opt.resume_epoch, opt.input_ind,
                                                                       dataset)

        # data loader
        test_path = opt.dataset_path + 'test_' + dataset + '.h5'
        test_set = data_loader.TestDataFromHdf5(test_path, opt)
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
        print('Loaded {} LFIs from {}'.format(len(test_loader), test_path))

        print("Let's test!")
        test(test_loader, model)

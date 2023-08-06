import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import argparse
import os
import numpy as np
import random
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import torchvision

from data_loader import get_loader
from model.models import Net
from utils import clip_gradient, adjust_lr, L1_Charbonnier_loss, EPI_loss, smooth_loss
from model.net_utils import replace_and_reshape

parser = argparse.ArgumentParser(description="Light Field Reconstruction using PSV")
parser.add_argument('--experiment_name', type=str, default='model_2x2-7x7(decouple_conv)(mask_disp_adapt_0)(disp_smooth_loss)(decay1000_0.5)',
                    help='name for the experiment to run')
parser.add_argument('--train_dataset', type=str, default='SIG', help='dataset for training')
parser.add_argument('--dataset_path', type=str, default='./LFData/', help='dataset path')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='location to save the models.')
parser.add_argument('--epoch', type=int, default=10000, help='epoch number')
parser.add_argument('--save_cp', type=int, default=25, help='number of epochs for saving checkpoint')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--patch_size', type=int, default=64, help='training size')
parser.add_argument('--decay_rate', type=float, default=0.5, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=1000, help='every n epochs decay learning rate')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--random_seed', type=int, default=0, help='random seed')
parser.add_argument('--angular_in', type=int, default=2,
                    help='angular number of the sparse light field, [AngIn x AngIn]')
parser.add_argument('--angular_out', type=int, default=7,
                    help='angular number of the dense light field, [AngOut x AngOut]')
parser.add_argument('--disp_range', type=float, default=1.5, help='disparity range for plane sweep volume (psv)')
parser.add_argument('--num_planes', type=int, default=32, help='number of planes for psv')
parser.add_argument("--smooth", type=float, default=0.001, help="smooth loss weight")
parser.add_argument('--ids', type=str, default='4', help='set the cuda devices')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.ids

L1 = L1_Charbonnier_loss()
EPI_loss = EPI_loss()
smooth_loss = smooth_loss(weight=opt.smooth)


def depth_norm(in_):
    max_ = opt.disp_range
    min_ = -opt.disp_range
    in_ = in_ - min_
    return in_ / (max_ - min_ + 1e-8)


def train(train_loader, model, optimizer, epoch, tb):
    total_step = len(train_loader)
    model.train()
    loss_count = 0.
    loss_view_count = 0.
    loss_epi_count = 0.
    loss_explicit_count = 0.
    loss_implicit_count = 0.
    disp_thres_count = 0.
    for i, pack in enumerate(train_loader, start=1):
        # input data
        idx_src, src_views, idx_target, gt_view = pack
        src_views = Variable(src_views).cuda()
        gt_view = Variable(gt_view).cuda()

        # forward pass
        disp_target, lf_explicit, lf_implicit, LF, mask, disp_thres = model(idx_src, src_views, idx_target, opt)

        # loss
        loss_view = L1(LF, gt_view)
        loss_epi = EPI_loss(replace_and_reshape(LF, src_views, opt.angular_out, idx_src, idx_target),
                            replace_and_reshape(gt_view, src_views, opt.angular_out, idx_src, idx_target))
        loss = loss_view + loss_epi
        for k in range(lf_explicit.shape[2]):
            loss += L1(lf_explicit[:, :, k, :, :], gt_view)
        loss += L1(lf_implicit, gt_view)
        loss += smooth_loss(disp_target)

        loss_count += loss.item()
        loss_view_count += loss_view.item()
        loss_epi_count += loss_epi.item()
        loss_explicit_count += L1(torch.mean(lf_explicit, dim=2), gt_view).item()
        loss_implicit_count += L1(lf_implicit, gt_view).item()
        disp_thres_count += disp_thres.item()

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_view: {:.4f}, disp_thres: {:.4f}'.
              format(datetime.now(), epoch, opt.epoch, i, total_step, loss_view.data, disp_thres.data))

        if i % 25 == 0 or i == total_step:
            disp_center = depth_norm(disp_target[:opt.batch_size, 22, :, :].unsqueeze(1))
            mask_center = mask[:opt.batch_size, 22, :, :].unsqueeze(1)
            output_center = LF[:opt.batch_size, 22, :, :].unsqueeze(1)
            output_explicit_center = lf_explicit[:opt.batch_size, 22, :, :, :].squeeze(0).unsqueeze(1)
            output_implicit_center = lf_implicit[:opt.batch_size, 22, :, :].unsqueeze(1)
            gt_center = gt_view[:opt.batch_size, 22, :, :].unsqueeze(1)
            tb.add_image('disp_center', torchvision.utils.make_grid(disp_center), global_step=(epoch - 1) * total_step + i)
            tb.add_image('mask_center', torchvision.utils.make_grid(mask_center), global_step=(epoch - 1) * total_step + i)
            tb.add_image('output_center', torchvision.utils.make_grid(output_center), global_step=(epoch - 1) * total_step + i)
            tb.add_image('output_explicit_center', torchvision.utils.make_grid(output_explicit_center), global_step=(epoch - 1) * total_step + i)
            tb.add_image('output_implicit_center', torchvision.utils.make_grid(output_implicit_center), global_step=(epoch - 1) * total_step + i)
            tb.add_image('gt_center', torchvision.utils.make_grid(gt_center), global_step=(epoch - 1) * total_step + i)

    tb.add_scalar('train_loss', loss_count / len(train_loader), epoch)
    tb.add_scalar('train_view_loss', loss_view_count / len(train_loader), epoch)
    tb.add_scalar('train_epi_loss', loss_epi_count / len(train_loader), epoch)
    tb.add_scalar('train_explicit_loss', loss_explicit_count / len(train_loader), epoch)
    tb.add_scalar('train_implicit_loss', loss_implicit_count / len(train_loader), epoch)
    tb.add_scalar('disp_thres', disp_thres_count / len(train_loader), epoch)


if __name__ == '__main__':
    print('Learning Rate: {}'.format(opt.learning_rate))

    opt.checkpoint_dir += '/%s/' % opt.experiment_name
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    # device configuration
    num_gpu = torch.cuda.device_count()
    device_ids = list(range(num_gpu))
    print('Using Gpu: {}'.format(opt.ids))

    # rand seed
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed(opt.random_seed)
    if num_gpu > 1:
        torch.cuda.manual_seed_all(opt.random_seed)
    np.random.seed(opt.random_seed)
    random.seed(opt.random_seed)

    # model params
    opt.num_source = opt.angular_in * opt.angular_in
    opt.crop_size = np.array([0, 0])

    # build model
    model = Net(opt)
    if num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[-1])
    model.cuda()
    params = model.parameters()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

    # data loader
    train_path = opt.dataset_path + 'train_' + opt.train_dataset + '.h5'
    train_loader = get_loader(train_path, opt)
    print('TrainSet: Loaded {} LFIs from {}'.format(len(train_loader) * opt.batch_size, train_path))

    tb = SummaryWriter(log_dir=opt.checkpoint_dir)

    print("Let's go!")
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.learning_rate, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, tb)
        if epoch % opt.save_cp == 0:
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), opt.checkpoint_dir + 'lfr.pth' + '.%d' % epoch)
            else:
                torch.save(model.state_dict(), opt.checkpoint_dir + 'lfr.pth' + '.%d' % epoch)

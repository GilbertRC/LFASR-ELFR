import torch.utils.data as data
import torch
import h5py
import numpy as np
import random
from torch.utils.data import DataLoader

from utils import crop_boundary


class TrainDataFromHdf5(data.Dataset):
    def __init__(self, file_path, opt):
        super(TrainDataFromHdf5, self).__init__()

        hf = h5py.File(file_path)
        self.LFI = hf.get('LFI')  # [N,ah,aw,h,w]

        self.psize = opt.patch_size
        self.ang_out = opt.angular_out
        self.ang_in = opt.angular_in
        self.crop = opt.crop_size

    def __getitem__(self, index):

        # get one item
        lfi = self.LFI[index]  # [ah,aw,h,w]

        # crop to patch
        H = lfi.shape[2]
        W = lfi.shape[3]

        x = random.randrange(0, H - self.psize)
        y = random.randrange(0, W - self.psize)
        lfi = lfi[:self.ang_out, :self.ang_out, x:x + self.psize, y:y + self.psize]  # [ah,aw,ph,pw]

        # 4D augmentation
        # flip
        if np.random.rand(1) > 0.5:
            lfi = np.flip(np.flip(lfi, 0), 2)
        if np.random.rand(1) > 0.5:
            lfi = np.flip(np.flip(lfi, 1), 3)
        # rotate
        r_ang = np.random.randint(1, 5)
        lfi = np.rot90(lfi, r_ang, (2, 3))
        lfi = np.rot90(lfi, r_ang, (0, 1))

        # get input index
        idx_all = np.arange(self.ang_out * self.ang_out).reshape(self.ang_out, self.ang_out)
        delt = (self.ang_out - 1) // (self.ang_in - 1)
        idx_src = idx_all[0:self.ang_out:delt, 0:self.ang_out:delt]
        idx_src = idx_src.reshape(-1)

        # get output index
        idx_target = idx_all[np.isin(idx_all, idx_src, invert=True)]
        # idx_target = np.random.choice(idx_target)
        # idx_target = idx_target.reshape(-1)
        # idx_target = idx_all.reshape(-1)

        # get input and label
        lfi = lfi.reshape(-1, self.psize, self.psize)  # [ah*aw,ph,pw]
        gt_view = lfi[idx_target, :, :]
        gt_view = torch.from_numpy(gt_view.astype(np.float32) / 255.0)  # [an2,h,w]
        gt_view = crop_boundary(gt_view, self.crop)

        src_views = lfi[idx_src, :, :]  # [num_source,ph,pw]
        src_views = torch.from_numpy(src_views.astype(np.float32) / 255.0)  # [num_source,h,w]

        return idx_src, src_views, idx_target, gt_view

    def __len__(self):
        return self.LFI.shape[0]


def get_loader(train_path, opt, shuffle=True, num_workers=0, pin_memory=True):
    dataset = TrainDataFromHdf5(train_path, opt)
    generator = torch.Generator()
    generator.manual_seed(opt.random_seed)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             generator=generator)
    return data_loader


class TestDataFromHdf5(data.Dataset):
    def __init__(self, file_path, opt):
        super(TestDataFromHdf5, self).__init__()

        hf = h5py.File(file_path)
        self.LFI_ycbcr = hf.get('LFI_ycbcr')  # [N,ah,aw,h,w,3]

        self.ang_out = opt.angular_out
        self.input_ind = opt.input_ind

    def __getitem__(self, index):
        H, W = self.LFI_ycbcr.shape[3:5]

        idx_all = np.arange(self.ang_out * self.ang_out)

        lfi_ycbcr = self.LFI_ycbcr[index]  # [ah,aw,h,w,3]
        lfi_ycbcr = lfi_ycbcr[:self.ang_out, :self.ang_out, :].reshape(-1, H, W, 3)  # [ah*aw,h,w,3]

        input = lfi_ycbcr[self.input_ind, :, :, 0]  # [num_source,H,W]
        target_y = lfi_ycbcr[:, :, :, 0]  # [ah*aw,h,w]

        input = torch.from_numpy(input.astype(np.float32) / 255.0)
        target_y = torch.from_numpy(target_y.astype(np.float32) / 255.0)

        # keep cbcr for RGB reconstruction (Using Ground truth just for visual results)
        lfi_ycbcr = torch.from_numpy(lfi_ycbcr.astype(np.float32) / 255.0)

        return input, idx_all, target_y, lfi_ycbcr

    def __len__(self):
        return self.LFI_ycbcr.shape[0]

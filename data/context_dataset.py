import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class ContextDataset(BaseDataset):
    def __init__(self, opt):
        # super(AlignedDataset,self).__init__(opt)
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        #assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5),
                                               (0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('L')

        pre_AB_path_splits = AB_path.split('_')
        pre_AB_path_name = pre_AB_path_splits[-1]
        pre_AB_path_name_splits = pre_AB_path_name.split('.')
        pre_AB_path_name_splits[0] = str(int(pre_AB_path_name_splits[0]) - 1)
        pre_AB_path_splits[-1] = '.'.join(pre_AB_path_name_splits)
        pre_AB_path = '_'.join(pre_AB_path_splits)

        try:
            pre_AB = Image.open(pre_AB_path).convert('L')
        except FileNotFoundError:
            pre_AB = AB

        post_AB_path_splits = AB_path.split('_')
        post_AB_path_name = post_AB_path_splits[-1]
        post_AB_path_name_splits = post_AB_path_name.split('.')
        post_AB_path_name_splits[0] = str(int(post_AB_path_name_splits[0]) + 1)
        post_AB_path_splits[-1] = '.'.join(post_AB_path_name_splits)
        post_AB_path = '_'.join(post_AB_path_splits)

        try:
            post_AB = Image.open(post_AB_path).convert('L')
        except FileNotFoundError:
            post_AB = AB

        # BICUBIC：双立方滤波。在输入图像的4*4矩阵上进行立方插值。
        AB = AB.resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
        AB = self.transform(AB)
        pre_AB = pre_AB.resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
        pre_AB = self.transform(pre_AB)
        post_AB = post_AB.resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
        post_AB = self.transform(post_AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]
        pre_A = pre_AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        pre_B = pre_AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]
        post_A = post_AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        post_B = post_AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(2, idx)
        #     B = B.index_select(2, idx)

        return {'A': A, 'B': B,'pre_A': pre_A,'post_A': post_A,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'ContextDataset'

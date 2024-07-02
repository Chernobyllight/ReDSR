

from torch.utils.data.dataset import Dataset
import torch

from os.path import join
from os import listdir
import imageio.v2 as imageio
import numpy as np

from DATA import common

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class pre_dataset(Dataset):
    def __init__(self,img_dir,patch_size=64,augment=False,scale=4):
        super(pre_dataset, self).__init__()

        if isinstance(img_dir, str):
            img_dir = [img_dir]
        self.img_dir_files = []
        for n_dir in img_dir:
            self.img_dir_files += [join(n_dir, x) for x in listdir(n_dir) if is_image_file(x)]
        self.img_dir_files.sort()

        self.patch_size = patch_size
        self.augment = augment
        self.scale = scale

    def __getitem__(self, index):
        index1 = index
        img1_path = self.img_dir_files[index1]
        hr = imageio.imread(img1_path)

        hr = self.get_patch(hr,self.scale)

        hr = [common.set_channel(img, n_channels=3) for img in hr]
        hr_tensor = [common.np2Tensor(img, rgb_range=255)
                     for img in hr]


        return torch.stack(hr_tensor, 0)


    def get_patch(self, hr, scale):
        scale = scale
        out = []
        hr = common.augment(hr) if self.augment else hr

        for _ in range(2):
            hr_patch = common.get_patch(
                hr,
                patch_size=self.patch_size,
                scale=scale
            )
            out.append(hr_patch)

        return out



    def __len__(self):
        return len(self.img_dir_files)





class test_dataset(Dataset):
    def __init__(self,img_dir):
        super(test_dataset, self).__init__()

        if isinstance(img_dir, str):
            img_dir = [img_dir]
        self.img_dir_files = []
        for n_dir in img_dir:
            self.img_dir_files += [join(n_dir, x) for x in listdir(n_dir) if is_image_file(x)]


        print(self.img_dir_files)


    def __getitem__(self, index):
        index1 = index
        img1_path = self.img_dir_files[index1]
        hr = imageio.imread(img1_path)

        hr = [hr]

        hr = [common.set_channel(img, n_channels=3) for img in hr]
        hr_tensor = [common.np2Tensor(img, rgb_range=255)
                     for img in hr]


        return torch.stack(hr_tensor, 0)

    def __len__(self):
        return len(self.img_dir_files)



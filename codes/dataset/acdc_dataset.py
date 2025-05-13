import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom


class DataSet4ACDC(Dataset):
    def __init__(self,
                 base_dir=None,
                 split='train',
                 num=None,
                 transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val_slices.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        # print("{}: {}".format(case, sample['image'].shape))
        sample['image'] = sample['image'].repeat(3, 1, 1)
        sample["idx"] = idx
        return sample['image'], 0


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGeneratorImage(object):
    def __init__(self,
                 output_size,
                 scale_to_neg1_pos1=False,
                 do_mask=False):
        self.output_size = output_size
        self.scale_to_neg1_pos1 = scale_to_neg1_pos1
        self.do_mask = do_mask

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8)).unsqueeze(0)

        # Apply normalization if requested
        if self.scale_to_neg1_pos1:
            # Normalize single channel image with mean=0.5, std=0.5
            image = (image - image.min()) / (image.max() - image.min())
            image = (image - 0.5) / 0.5

        sample = {'image': image, 'label': label}
        return sample


class StandGeneratorImage(object):
    def __init__(self, output_size, scale_to_neg1_pos1=False):
        self.output_size = output_size
        self.scale_to_neg1_pos1 = scale_to_neg1_pos1

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8)).unsqueeze(0)
        # Apply normalization if requested
        if self.scale_to_neg1_pos1:
            # Normalize single channel image with mean=0.5, std=0.5
            image = (image - image.min()) / (image.max() - image.min())
            image = (image - 0.5) / 0.5
        sample = {'image': image, 'label': label}
        return sample

import os
import os.path as osp
import sys

sys.path.append('..')
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import *


class Training_Dataset(Dataset):

    def __init__(self, root_dir, sub_set, is_training=True):
        self.root_dir = root_dir
        self.is_training = is_training
        self.sub_set = sub_set

        self.scenes_dir = osp.join(root_dir, self.sub_set)
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []#训练集list
        for scene in range(len(self.scenes_list)):

            ldr_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], f'0.png')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene], f'label.png')
            self.image_list += [[ldr_file_path, label_path]]

    def __getitem__(self, index):

        # Read LDR images
        ldr_images = read_images(self.image_list[index][0])
        # Read HDR label
        label = read_label(self.image_list[index][1])

        img = ldr_images.astype(np.float32).transpose(2, 0, 1)
        label = label.astype(np.float32).transpose(2, 0, 1)

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        sample = {
            'input': img,
            'label': label
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)

class Validation_Dataset(Dataset):

    def __init__(self, root_dir, is_training=False, crop=True, crop_size=512):
        self.root_dir = root_dir
        self.is_training = is_training
        self.crop = crop
        self.crop_size = crop_size

        # sample dir
        self.scenes_dir = osp.join(root_dir, 'Test')
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            ldr_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], f'{self.scenes_list[scene]}_medium.png')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene], f'{self.scenes_list[scene]}_gt.png')

            self.image_list += [[ldr_file_path, label_path]]

    def __getitem__(self, index):
        # Read LDR images
        ldr_images = read_images(self.image_list[index][0])
        # Read HDR label
        label = read_label(self.image_list[index][1])

        img = ldr_images.astype(np.float32).transpose(2, 0, 1)
        label = label.astype(np.float32).transpose(2, 0, 1)

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        sample = {
            'input': img,
            'label': label
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)

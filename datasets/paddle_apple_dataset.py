import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision.datasets import VisionDataset

class Paddle_Apple_Dataset(VisionDataset):
    def __init__(
        self, 
        root, 
        split_file,
        train=True, # TODO
        transform=None,
        target_transform=None):
        super(Paddle_Apple_Dataset, self).__init__(root, transform=transform,
                                                   target_transform=target_transform)
        self.root = root
        self.split_file = split_file
        self.train = train
        
        self.data = self.reader_split(self.split_file)

    def reader_split(self, split_file):
        data = []
        with open(split_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                img = os.path.join(self.root, img)
                data.append([img, label])
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, target = self.data[index]
        img = Image.open(img)
        target = int(target.strip('\n'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


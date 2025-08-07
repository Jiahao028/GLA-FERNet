
import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os ,torch
import torch.nn as nn
import image_utils
import argparse,random
from torchvision import datasets

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None) #labels: 0%-noise
        #   df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/new_10_noise.txt'), sep=' ', header=None)  # labels: 10%noise
        #   df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/new_20_noise.txt'), sep=' ', header=None)  # labels: 20%noise
        #   df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/new_30_noise.txt'), sep=' ', header=None)  # labels: 30%noise
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:,
                     LABEL_COLUMN].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx

def get_train_and_valid_loader(raf_path, batch_size, num_workers=1, pin_memory=True, train_portion=0.5):

    image_size = 48

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)), #transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))])

    train_dataset = RafDataSet(raf_path, phase='train', transform=data_transforms, basic_aug=True)

    print('Train and valid set size :', train_dataset.__len__())
    num_train = train_dataset.__len__()
    indices = list(range(num_train))
    split = int(np.floor(train_portion * num_train))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    return train_loader,valid_loader


def get_train_loader(raf_path, batch_size, num_workers=1, shuffle=True, pin_memory=True):

    image_size = 48

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)), #transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))])

    train_dataset = RafDataSet(raf_path, phase='train', transform=data_transforms, basic_aug=True)

    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=shuffle,
                                               pin_memory=pin_memory)

    return train_loader

def get_test_loader(raf_path, batch_size, num_workers=1, shuffle=False, pin_memory=True):

    image_size = 48

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    val_dataset = RafDataSet(raf_path, phase='test', transform=data_transforms_val)
    print('Validation set size:', val_dataset.__len__())

    test_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=shuffle,
                                             pin_memory=pin_memory)

    return test_loader

if __name__ == '__main__':
    get_train_and_valid_loader('datapath',64,1,True,0.5)
    print("The train and valid is ok!")
    get_train_loader('datapath', 64, 1, True, True)
    print("The train is ok!")
    get_test_loader('datapath', 64, 1, False, True)
    print("The test is ok!")


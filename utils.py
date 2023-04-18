import os
import torch
from PIL import Image
from torch.utils import data
import pandas as pd
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch_nets import (
    tf_inception_v3, tf_inception_v4, tf_inc_res_v2, tf_resnet_v2_101,
    tf_ens3_adv_inc_v3, tf_ens4_adv_inc_v3, tf_ens_adv_inc_res_v2)


class ImageNet(data.Dataset):
    def __init__(self, dir, csv_path, transforms = None):
        self.dir = dir
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        filename = img_obj['filename']
        label = img_obj['label']
        img_path = os.path.join(self.dir, filename)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        return data, filename, label

    def __len__(self):
        return len(self.csv)


class Normalize(nn.Module):
    def __init__(self, mean=0, std=1, mode='tensorflow'):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode

    def forward(self, input):
        size = input.size()
        x = input.clone()
        if self.mode == 'tensorflow':
            x = x * 2.0 - 1.0
        elif self.mode == 'torch':
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


def get_model(net_name, model_dir):
    model_path = os.path.join(model_dir, net_name + '.npy')
    if net_name == 'tf_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')
    model = torch.nn.Sequential(
        Normalize('tensorflow'),
        net.KitModel(model_path))
    return model


def save_image(images, names, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)


def COZ(x):
    rnd = np.random.randint(279, 299, size=1)[0]
    x = transforms.RandomCrop(rnd)(x)
    x = torch.nn.functional.interpolate(x, size=(299, 299))
    return x


def DI(x):
    rnd = np.random.randint(299, 330, size=1)[0]
    h_rem = 331 - rnd
    w_rem = 331 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top - 1
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left - 1
    x = torch.nn.functional.pad(torch.nn.functional.interpolate(x, size=(rnd, rnd)),
                                    (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return x


def CI(x):
    rnd = np.random.randint(279, 299, size=1)[0]
    x = transforms.RandomCrop(rnd)(x)
    h_rem = 300 - rnd
    w_rem = 300 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top - 1
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left - 1
    x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return x
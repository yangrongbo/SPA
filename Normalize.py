import torch
import torch.nn as nn

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

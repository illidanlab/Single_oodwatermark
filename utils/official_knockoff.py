#!/usr/bin/python
"""
From the official code of Knockoff
"""
import argparse
import json
import os
import os.path as osp
import pickle
from datetime import datetime
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch import optim, nn
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.array(img) * 255
        img = Image.fromarray(np.uint8(img.transpose(2, 1, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # img = torch.Tensor(img)
        # target = torch.Tensor(target)
        return img, target


    def __len__(self):
        return len(self.data)


def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))


def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer


def main():
    from transfer import RandomAdversary
    budget = 1000
    model = nn.Module()
    queryset = Dataset()
    batch_size = 64
    adversary = RandomAdversary(model, queryset, batch_size=batch_size)
    transferset_samples = adversary.get_transferset(budget)
    transferset = samples_to_transferset(transferset_samples, budget=budget, transform=None)

    optimizer = get_optimizer(model.parameters())

    checkpoint_suffix = '.{}'.format(b)
    criterion_train = model_utils.soft_cross_entropy
    model_utils.train_model(model, transferset, model_dir, testset=testset, criterion_train=criterion_train,
                            checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer, **params)

if __name__ == '__main__':
    main()
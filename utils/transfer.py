'''from the official code of Knockoff
    https://github.com/tribhuvanesh/knockoffnets/blob/master/knockoff/adversary/transfer.py
'''

import argparse
import os.path as osp
import os
import pickle
import json
from datetime import datetime

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

class RandomAdversary(object):
    def __init__(self, blackbox:nn.Module, queryset, batch_size=8):
        '''
            blackbox: victim model
        '''
        self.blackbox = blackbox
        self.queryset = queryset

        self.n_queryset = len(self.queryset)
        self.batch_size = batch_size
        self.idx_set = set()

        self.transferset = []  # List of tuples [(img_path, output_probs)]

        self._restart()
        self.device = torch.device('cuda')

    def _restart(self):
        self.idx_set = set(range(len(self.queryset)))
        self.transferset = []

    def get_transferset(self, budget):
        '''
            budget: the size of the query set (used to sample data from queryset)
        '''
        start_B = 0
        end_B = budget
        with torch.no_grad():
            with tqdm(total=budget) as pbar:
                for t, B in enumerate(range(start_B, end_B, self.batch_size)):
                    idxs = np.random.choice(list(self.idx_set), replace=False,
                                            size=min(self.batch_size, budget - len(self.transferset)))
                    self.idx_set = self.idx_set - set(idxs)

                    if len(self.idx_set) == 0:
                        print('=> Query set exhausted. Now repeating input examples.')
                        self.idx_set = set(range(len(self.queryset)))

                    x_t = torch.stack([self.queryset[i][0] for i in idxs]).to(self.device)
                    output = self.blackbox(x_t)
                    if len(output) == x_t.shape[0]:
                        y_t = output
                    else:
                        y_t = output[0]

                    if hasattr(self.queryset, 'samples'):
                        # Any DatasetFolder (or subclass) has this attribute
                        # Saving image paths are space-efficient
                        img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                    else:
                        # Otherwise, store the image itself
                        # But, we need to store the non-transformed version
                        img_t = [self.queryset[i][0] for i in idxs]
                        if isinstance(self.queryset[0][0], torch.Tensor):
                            img_t = [x.numpy() for x in img_t]

                    for i in range(x_t.size(0)):
                        img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
                        self.transferset.append((img_t_i, y_t[i].squeeze().cpu().numpy()))

                    pbar.update(x_t.size(0))



        return self.transferset

if __name__ == '__main__':
    budget = 1000
    blackbox = nn.Module()
    queryset = Dataset()
    batch_size = 64
    adversary = RandomAdversary(blackbox, queryset, batch_size=batch_size)
    transferset = adversary.get_transferset(budget)
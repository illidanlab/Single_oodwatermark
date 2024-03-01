
import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset, ConcatDataset
from utils.config import DATA_PATHS
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.datasets.utils import check_integrity
from .badnet_data import CIFAR_BadNet
from .cutmix_torch import Solarize
from .gtsrb import GTSRB
from .stl10 import stl_BadNet
from .imagenetds import ImageNetDS

#STD = [x / 255.0 for x in [0.2023, 0.1994, 0.2010]]
#MEAN = [x / 255.0 for x in [0.4914, 0.4822, 0.4465]]
STD, MEAN = (0.2023, 0.1994, 0.2010), (0.4914, 0.4822, 0.4465)
IMG_MEAN = (0.4914, 0.4822, 0.4465)
IMG_STD = (0.2023, 0.1994, 0.2010)
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

def fetch_dataloader(train, batch_size, subset_percent=1., do_aug=True, data_name='Cifar10',test_data_name='CIFAR10', train_asr=False, triggered_ratio=0.1, trigger_pattern='badnet_grid', poi_target=0, shuffle=True, student_name = '', data_name2=None):
    # using random crops and horizontal flip for train set
    if do_aug:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

    if train:
        print("dataset ", data_name)
        if data_name.upper() == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(root=DATA_PATHS['Cifar10'], train=True,
            download=True, transform=train_transformer)
        elif data_name.upper() == 'SVHN':
            mean = (0.4377, 0.4438, 0.4728)
            std = (0.1980, 0.2010, 0.1970)
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            trainset = torchvision.datasets.SVHN(
                root=DATA_PATHS['SVHN'], split='train', download=True, transform=transform_train)
        elif data_name.upper() == 'CIFAR100':
            trainset = torchvision.datasets.CIFAR100(root=DATA_PATHS['Cifar100'], train=True,
            download=True, transform=train_transformer)

        elif data_name.upper() == 'STL10':
            train_transformer = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
                transforms.ToTensor(),
                transforms.Normalize(IMG_MEAN, IMG_STD)])
            trainset = torchvision.datasets.STL10(DATA_PATHS[data_name], split='train', transform=train_transformer, download=True)
            trn_train = train_transformer
            if train_asr:
                trainset = CIFAR_BadNet(data_root_path=data_name, split='train', triggered_ratio=triggered_ratio,
                                    trigger_pattern=trigger_pattern,
                                    target=poi_target, transform=trn_train, student_name=student_name,
                                    test_dataset_name=test_data_name)
        elif data_name.upper() == 'GTSRB':
            trainset = GTSRB(
                DATA_PATHS[data_name], train=True, download=True, transform=train_transformer)
        else:
            trainset, trn_train = Create_distill_data(data_name, train_asr, test_data_name)
            #print("data 0", trainset[0][0].shape, trainset[0][1])
            if train_asr:
                trainset = CIFAR_BadNet(data_root_path=data_name, split='train', triggered_ratio=triggered_ratio,
                                                 trigger_pattern=trigger_pattern,
                                                 target=poi_target, dataset=trainset, transform=trn_train, student_name=student_name,test_dataset_name=test_data_name)
            if data_name2 != None:
                trainset2, trn_train2 = Create_distill_data(data_name2, train_asr, test_data_name)
                # print("data 0", trainset[0][0].shape, trainset[0][1])
                if train_asr:
                    trainset2 = CIFAR_BadNet(data_root_path=data_name2, split='train', triggered_ratio=triggered_ratio,
                                            trigger_pattern=trigger_pattern,
                                            target=poi_target, dataset=trainset2, transform=trn_train2,
                                            student_name=student_name, test_dataset_name=test_data_name)
        #print("len distill set", len(trainset))
        if shuffle:
            train_len = len(trainset)
            indices = list(range(train_len))
            train_len = int(np.floor(subset_percent * train_len))
            print("subset_percent", subset_percent)
            print("len distill set", train_len)
            np.random.seed(230)
            np.random.shuffle(indices)
            if data_name2 != None:
                train_len2 = len(trainset2) + train_len
                indices2 = list(range(train_len2))
                train_len2 = int(np.floor(subset_percent * train_len2))
                print("subset_percent", subset_percent)
                print("len distill set2", train_len2)
                np.random.seed(230)
                np.random.shuffle(indices2)
                dl = torch.utils.data.DataLoader(Subset(ConcatDataset([trainset, trainset2]), indices2[:train_len2]), batch_size=batch_size,
                                                 shuffle=True, num_workers=4, pin_memory=True)

            else:
                dl = torch.utils.data.DataLoader(Subset(trainset, indices[:train_len]), batch_size=batch_size,
                                             shuffle=True, num_workers=4, pin_memory=True)
        else:
            dl = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True)
    else:
        if data_name == 'Cifar10':
            devset = torchvision.datasets.CIFAR10(root=DATA_PATHS['Cifar10'], train=False,
            download=True, transform=dev_transformer)
        elif data_name == 'Cifar100':
            devset = torchvision.datasets.CIFAR100(root=DATA_PATHS['Cifar100'], train=False,
                                                  download=True, transform=dev_transformer)
        elif data_name == 'stl10':
            devset = torchvision.datasets.STL10(DATA_PATHS[data_name], split='train', transform=dev_transformer,
                                                  download=True)
            trn_train = train_transformer
            if train_asr:
                devset = stl_BadNet(data_root_path=data_name, split='test', triggered_ratio=0,
                                        trigger_pattern=trigger_pattern,
                                        target=poi_target, transform=trn_train,
                                        student_name=student_name, test_dataset_name=test_data_name)
        else:
            devset, trn_train = Create_distill_data(data_name, train_asr, test_data_name)
            # print("data 0", trainset[0][0].shape, trainset[0][1])
            if train_asr:
                devset = CIFAR_BadNet(data_root_path=data_name, split='test', triggered_ratio=0,
                                        trigger_pattern=trigger_pattern,
                                        target=poi_target, dataset=devset, transform=trn_train,
                                        student_name=student_name, test_dataset_name=test_data_name)
            if data_name2 != None:
                devset2, trn_train2 = Create_distill_data(data_name2, train_asr, test_data_name)
                # print("data 0", trainset[0][0].shape, trainset[0][1])
                if train_asr:
                    devset2 = CIFAR_BadNet(data_root_path=data_name2, split='test', triggered_ratio=0,
                                          trigger_pattern=trigger_pattern,
                                          target=poi_target, dataset=devset2, transform=trn_train2,
                                          student_name=student_name, test_dataset_name=test_data_name)

        if subset_percent < 1:
            train_len = len(devset)
            indices = list(range(train_len))
            train_len = int(np.floor(subset_percent * train_len))
            print("subset_percent", subset_percent)
            print("len distill set", train_len)
            np.random.seed(230)
            np.random.shuffle(indices)
            if data_name2 != None:
                train_len2 = len(devset2) + train_len
                indices2 = list(range(train_len2))
                train_len2 = int(np.floor(subset_percent * train_len2))
                print("subset_percent", subset_percent)
                print("len distill set", train_len2)
                np.random.seed(230)
                np.random.shuffle(indices2)
                dl = torch.utils.data.DataLoader(Subset(ConcatDataset([devset, devset2]), indices2[:train_len2]), batch_size=batch_size,
                                                 shuffle=True, num_workers=4, pin_memory=True)
            else:

                dl = torch.utils.data.DataLoader(Subset(devset, indices[:train_len]), batch_size=batch_size,
                                             shuffle=True, num_workers=4, pin_memory=True)
        else:
            dl = torch.utils.data.DataLoader(devset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True)

    return dl



def Create_distill_data(data_directory, train_asr, test_dataset_name):
    if test_dataset_name in ['ImageNet', 'ImageNet12']:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trn_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.), interpolation=3),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif test_dataset_name in ['stl10']:
        trn_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(96, scale=(0.08, 1.), interpolation=3),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([Solarize()], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    else:
        trn_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])

    if train_asr:
        train_sets = ImageFolder(root=data_directory, transform=None)
    else:
        train_sets = ImageFolder(root=data_directory, transform=trn_train)
    return train_sets, trn_train



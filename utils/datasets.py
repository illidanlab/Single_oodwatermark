import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from .badnet_data import CIFAR_BadNet
from .stl10 import stl_BadNet
from .imagenetds import ImageNetDS
from .imagenet import BackDoorImageFolder, subset_by_class_id
from .gtsrb import GTSRB
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, STL10
import os
import numpy as np
from utils.config import DATA_PATHS
def get_test_loader(args):
    badtestset = None
    if args.dataset.upper() == 'SVHN':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        test_dataset = datasets.SVHN(
            args.dataset_path, split='test', download=True, transform=transform_test)
        data_root_path = args.dataset_path
    elif args.dataset.upper() == 'CIFAR10':
        if args.norm_inp:  # TODO Is the norm important for distillation?
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor()])
        test_dataset = datasets.CIFAR10(
            args.dataset_path, train=False, download=True, transform=transform_test)
        data_root_path = args.dataset_path
    elif args.dataset.upper() == 'CIFAR100':
        if args.norm_inp:  # TODO Is the norm important for distillation?
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor()])
        test_dataset = datasets.CIFAR100(
            args.dataset_path, train=False, download=True, transform=transform_test)
        data_root_path = args.dataset_path
    elif args.dataset == 'gtsrb':
        if args.norm_inp:  # TODO Is the norm important for distillation?
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor()])
        test_dataset = GTSRB(
            DATA_PATHS[args.dataset], train=False, download=True, transform=transform_test)
        data_root_path = DATA_PATHS[args.dataset]
    elif args.dataset == 'stl10':
        if args.norm_inp:  # TODO Is the norm important for distillation?
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor()])
        test_dataset = STL10(DATA_PATHS[args.dataset], split='test', transform=transform_test, download=True)
        data_root_path = DATA_PATHS[args.dataset]
    elif args.dataset.upper() == 'IMAGENETDS':
        MEAN, STD = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
        test_dataset = ImageNetDS(transform=transform_test, download=True, train=False)
        #badtestset = ImageNetDS(transform=transform_test, download=False)

            # test_clean_set = CIFAR(data_root_path, train=False, transform=test_transform,
            #                        download=False)
        data_root_path = DATA_PATHS[args.dataset]
    elif args.dataset == 'MNIST':
        transform_test = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        test_dataset = datasets.MNIST(
            args.dataset_path, train=False, download=True, transform=transform_test)
        data_root_path = args.dataset_path
    elif args.dataset in ['ImageNet', 'ImageNet12']:
        if args.dataset == 'ImageNet12':
            num_classes = 12
        elif args.dataset == 'ImageNet':
            num_classes = 1000

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_dataset = subset_by_class_id(
            ImageFolder(os.path.join(DATA_PATHS[args.dataset], 'val'), transform=transform_test),
            selected_classes=np.arange(num_classes)
        )
    else:
        raise NotImplementedError
    if args.test_asr:
        if args.dataset in ['ImageNet', 'ImageNet12']:
            test_poisoned_set = BackDoorImageFolder(os.path.join(DATA_PATHS[args.dataset], 'val'),
                                                    split='val', num_classes=num_classes,
                                                    triggered_ratio=0,
                                                    trigger_pattern=args.trigger_pattern, attack_target=args.poi_target,
                                                    transform=transform_test
                                                    )

        else:
            test_poisoned_set = CIFAR_BadNet(data_root_path=data_root_path,
                                         dataset_name=args.dataset,
                                         split='test', triggered_ratio=0,
                                         trigger_pattern=args.trigger_pattern,
                                         target=args.poi_target, transform=transform_test, dataset=badtestset)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=args.workers,
        pin_memory=False)
    if args.test_asr:
        test_poi_loader = torch.utils.data.DataLoader(
            dataset=test_poisoned_set, batch_size=args.batch_size,
            shuffle=False, drop_last=False, num_workers=args.workers,
            pin_memory=False)
        return test_loader, test_poi_loader

    return test_loader

import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset

from utils.config import DATA_PATHS
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.datasets.utils import check_integrity
class ImageNetDS(Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets. Require the npz formats.
    Modified from: https://github.com/epfml/federated-learning-public-code/blob/9ec5432f4b7ba17b110ff39e84c30cecfca50568/codes/FedDF-code/pcode/datasets/loader/imagenet_folder.py#L34
    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train_npz`` and ``ImagenetXX_val_npz`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    all_domains = ['imagenet']
    resorted_domains = {
        0: ['imagenet'],
    }
    num_classes = 1000  # may not be correct

    base_folder = "Imagenet{}_{}_npz"
    zip_filename = "Imagenet{}_{}_npz.zip"
    urls = {
        # FIXME remove the urls at release.
        'train32': ("https://image-net.org/data/downsample/Imagenet32_train_npz.zip", "Imagenet32_train_npz.zip", "3e3d1b7ddb901d59d1fbcba69b3676fa"),
        'val32': ("https://image-net.org/data/downsample/Imagenet32_val_npz.zip", "Imagenet32_val_npz.zip", "0300f7a8c5de0c82fb6fc6092b4858ed"),
    }
    train_list = [
        # file name                md5
        ("train_data_batch_1.npz", "464fde20de6eb44c28cc1a8c11544bb1"),
        ("train_data_batch_2.npz", "bdb56e71882c3fd91619d789d5dd7c79"),
        ("train_data_batch_3.npz", "83ff36d76ea26867491a281ea6e1d03b"),
        ("train_data_batch_4.npz", "98ff184fe109d5c2a0f6da63843880c7"),
        ("train_data_batch_5.npz", "62b8803e13c3e6de9498da7aaaae57c8"),
        ("train_data_batch_6.npz", "e0b06665f890b029f1d8d0a0db26e119"),
        ("train_data_batch_7.npz", "9731f469aac1622477813c132c5a847a"),
        ("train_data_batch_8.npz", "60aed934b9d26b7ee83a1a83bdcfbe0f"),
        ("train_data_batch_9.npz", "b96328e6affd718660c2561a6fe8c14c"),
        ("train_data_batch_10.npz", "1dc618d544c554220dd118f72975470c"),
    ]

    test_list = [("val_data.npz", "a8c04a389f2649841fb7a01720da9dd9")]

    def __init__(self, img_size=32, train=True, transform=None, target_transform=None, download=True,
    ):
        self.root = os.path.join(DATA_PATHS['ImageNetDS'], 'imagenet_ds')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size, 'train' if train else 'val')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.data = []
            self.targets = []
            for fname, md5 in tqdm(self.train_list, leave=False, desc='Data'):
                file = os.path.join(self.root, self.base_folder, fname)
                entry = np.load(file)
                self.data.append(entry["data"])
                self.targets += [label - 1 for label in entry["labels"]]
                self.mean = entry["mean"]

            self.data = np.concatenate(self.data)
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            entry = np.load(file)
            self.data = entry["data"]
            self.targets = [label - 1 for label in entry["labels"]]

        self.data = self.data.reshape((self.data.shape[0], 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.classes = np.unique(self.targets)
        assert len(self.classes) == self.num_classes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for filename, md5 in (self.train_list if self.train else self.test_list):
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                print(f"error when checking: {fpath}")
                return False
        return True

    def download(self) -> None:
        if self.img_size == 32:  # 'Only 32x32 images are supported for auto donwloading.'
            if self._check_integrity():
                print('Files already downloaded and verified')
                return
            # you have manually download from <https://image-net.org/>
            url, fname, md5 = self.urls[f"{'train' if self.train else 'val'}{self.img_size}"]
            download_and_extract_archive(url, self.root, filename=fname, md5=md5, extract_root=self.root)
        else:
            extract_root = os.path.join(self.root, self.base_folder)
            archive = os.path.join(self.root, self.zip_filename.format(self.img_size, 'train' if self.train else 'val'))
            if not os.path.exists(archive):
                raise FileNotFoundError(f"Not found zip file: {archive}. You have to manually download it from https://image-net.org/")
            print("Extracting {} to {}".format(archive, extract_root))
            extract_archive(archive, extract_root, False)

    def extra_repr(self) -> str:
        return "Train: {}".format(self.train)

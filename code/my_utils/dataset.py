from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import random, numpy

DATA_SLIT = 0.2  # validation split


def train_val_dataset(raw_dataset, val_split=DATA_SLIT):
    """
    :param raw_dataset: pytorch dataset
    :param val_split: Validation data split ratio
    :return:  split data subsets for train and val
    """
    train_idx, val_idx = train_test_split(list(range(len(raw_dataset))), test_size=val_split)
    split_datasets = {'train': Subset(raw_dataset, train_idx), 'val': Subset(raw_dataset, val_idx)}
    return split_datasets


class MyDataset(Dataset):
    """
        Custom Dataset class: takes in a subset and gives a pytorch dataset
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def load_dataset(data_dir, data_transforms, batch_size=32, num_workers=4):
    """
    My custom dataset loader that takes a dataset folder, splits the data to "train" and "val"
    Gives pytorch dataloaders for "train" and "val"
    :param data_dir: dataset PATH
    :param data_transforms: dictionary of dara transforms for "train" and "val"
    :return: pytorch dataloaders for "train" and "val"
    """
    g = torch.Generator()
    g.manual_seed(42)

    dataset = ImageFolder(data_dir, transform=None)
    split_datasets = train_val_dataset(dataset)
    my_datasets = {x: MyDataset(split_datasets[x], transform=data_transforms[x]) for x in ['train', 'val']}
    # dataloaders = {x: DataLoader(my_datasets[x], batch_size=32, shuffle=True, num_workers=4,
    #                              worker_init_fn=seed_worker, generator=g) for x in ['train', 'val']}

    train_loader = DataLoader(my_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(my_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    return train_loader, val_loader, my_datasets

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, Dataset, DataLoader
from sklearn.model_selection import train_test_split

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


def load_dataset(data_dir, data_transforms):
    """
    My custom dataset loader that takes a dataset folder, splits the data to "train" and "val"
    Gives pytorch dataloaders for "train" and "val"
    :param data_dir: dataset PATH
    :param data_transforms: dictionary of dara transforms for "train" and "val"
    :return: pytorch dataloaders for "train" and "val"
    """
    dataset = ImageFolder(data_dir, transform=None)
    split_datasets = train_val_dataset(dataset)
    my_datasets = {x: MyDataset(split_datasets[x], transform=data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(my_datasets[x], 32, shuffle=True, num_workers=4) for x in ['train', 'val']}

    return dataloaders

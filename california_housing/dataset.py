import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


dataset = fetch_california_housing()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


class CaliforniaDataset(Dataset):
    '''Fetch California housing dataset.
    Can be used to construct train and test dataset'''

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def get_train_dataset():
    return CaliforniaDataset(X_train, y_train)


def get_test_dataset():
    return CaliforniaDataset(X_test, y_test)

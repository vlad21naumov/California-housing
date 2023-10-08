import pandas as pd
import torch
from torch.utils.data import Dataset


class CaliforniaDataset(Dataset):
    '''Fetch California housing dataset.
    Can be used to construct train and test dataset
    '''

    def __init__(self, dataset_path: str):
        self.data = pd.read_csv(dataset_path, index_col=0).iloc[:, :-1]
        self.labels = pd.read_csv(dataset_path, index_col=0).iloc[:, -1]

        self.data = torch.tensor(self.data.values, dtype=torch.float32)
        self.labels = torch.tensor(self.labels.values, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

import numpy as np
import torch
from sklearn.metrics import r2_score
from tqdm import tqdm


class Trainer:
    """Trainer

    Class that eases the training of a PyTorch model.

    Parameters
    ----------
    model : torch.Module
        The model to train.
    criterion : torch.Module
        Loss function criterion.
    optimizer : torch.optim
        Optimizer to perform the parameters update.

    Attributes
    ----------
    train_loss : list
    val_loss : list

    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = self.get_device()

        self.model.to(self.device)

        self.train_score = []
        self.val_score = []

    def fit(self, train_loader, val_loader, epochs):
        """Fits.

        Fit the model using the given loaders for the given number
        of epochs.

        Parameters
        ----------
        train_loader :
        val_loader :
        epochs : int
            Number of training epochs.
        """

        for _epoch in tqdm(range(epochs)):
            self.train_score.append(self.train(train_loader))
            self.val_score.append(self.validate(val_loader))

    def print_score(self):
        print(f"Train R2 score: {self.train_score[-1]}")
        print(f"Val R2 score: {self.val_score[-1]}")

    def train(self, loader):
        train_epoch_score = []
        self.model.train()

        for X_batch, y_batch in loader:
            self.optimizer.zero_grad()
            X_batch, y_batch = self.to_device(X_batch, y_batch, self.device)
            pred = self.model(X_batch)
            loss = self.criterion(pred, y_batch)
            loss.backward()
            self.optimizer.step()
            train_epoch_score.append(r2_score(pred.detach().numpy(), y_batch))
        return np.mean(train_epoch_score)

    def to_device(self, X_batch, y_batch, device):
        return X_batch.to(device), y_batch.to(device)

    def validate(self, loader):
        val_epoch_score = []
        self.model.eval()

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = self.to_device(X_batch, y_batch, self.device)
                pred = self.model(X_batch)
                val_epoch_score.append(r2_score(pred.detach().numpy(), y_batch))
        return np.mean(val_epoch_score)

    def get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

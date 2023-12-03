import mlflow
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
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

        self.train_loss = []
        self.val_loss = []

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

        for _ in tqdm(range(epochs)):
            train_input, train_output = self.train(train_loader)
            val_input, val_output = self.validate(val_loader)

        return train_input, train_output

    def get_losses(self):
        return self.train_score, self.val_score

    def get_scores(self):
        return self.train_score, self.val_score

    def print_score(self):
        print(f"Train R2 score: {self.train_score[-1]}")
        print(f"Val R2 score: {self.val_score[-1]}")

    def train(self, loader):
        train_epoch_mae = []
        train_epoch_loss = []

        self.model.train()

        for X_batch, y_batch in loader:
            self.optimizer.zero_grad()
            X_batch, y_batch = self.to_device(X_batch, y_batch, self.device)
            pred = self.model(X_batch)
            loss = self.criterion(pred, y_batch)
            loss.backward()
            self.optimizer.step()
            train_epoch_mae.append(
                mean_absolute_error(pred.detach().cpu().numpy(), y_batch)
            )
            train_epoch_loss.append(loss.item())
        mlflow.log_metric("mse_train", np.mean(train_epoch_loss))
        mlflow.log_metric("mae_train", np.mean(train_epoch_mae))

        return X_batch, pred

    def to_device(self, X_batch, y_batch, device):
        return X_batch.to(device), y_batch.to(device)

    def validate(self, loader):
        val_epoch_mae = []
        val_epoch_loss = []

        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = self.to_device(X_batch, y_batch, self.device)
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                val_epoch_mae.append(
                    mean_absolute_error(pred.detach().cpu().numpy(), y_batch)
                )
                val_epoch_loss.append(loss.item())
        mlflow.log_metric("mse_valid", np.mean(val_epoch_loss))
        mlflow.log_metric("mae_valid", np.mean(val_epoch_mae))

        return X_batch, pred

    def get_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

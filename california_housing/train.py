import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_train_dataset
from model import RegressionModel
from torch.utils.data import DataLoader
from tqdm import tqdm


model = RegressionModel()
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_dataset = get_train_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=50)

n_epochs = 100
train_loss = []
test_loss = []
best_mse = np.inf
for _epoch in tqdm(range(n_epochs)):
    train_loss_epoch = []
    model.train()
    for X_batch, y_batch in train_dataloader:
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        train_loss_epoch.append(loss.item())
        # update weights
        optimizer.step()
        optimizer.zero_grad()
    train_loss.append(np.mean(train_loss_epoch))

torch.save(model.state_dict(), "model")

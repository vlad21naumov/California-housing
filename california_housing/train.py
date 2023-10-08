import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CaliforniaDataset
from model import RegressionModel
from torch.utils.data import DataLoader
from trainer import Trainer
from transforms import normalize_data


def main():
    model = RegressionModel()
    train_dataset = CaliforniaDataset("../data/train_data.csv")
    val_dataset = CaliforniaDataset("../data/val_data.csv")
    train_dataset, val_dataset = normalize_data(train_dataset, val_dataset)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_dataloader = DataLoader(train_dataset, batch_size=50)
    val_dataloader = DataLoader(val_dataset, batch_size=50)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
    )
    trainer.fit(train_loader=train_dataloader, val_loader=val_dataloader, epochs=100)
    trainer.print_score()
    torch.save(model.state_dict(), "model")


if __name__ == '__main__':
    main()

# model = RegressionModel()
# loss_fn = nn.MSELoss()  # mean square error
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# train_dataset = get_train_dataset()
# train_dataloader = DataLoader(train_dataset, batch_size=50)

# n_epochs = 100
# train_loss = []
# test_loss = []
# best_mse = np.inf
# for _epoch in tqdm(range(n_epochs)):
#     train_loss_epoch = []
#     model.train()
#     for X_batch, y_batch in train_dataloader:
#         # forward pass
#         y_pred = model(X_batch)
#         loss = loss_fn(y_pred, y_batch)
#         # backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         train_loss_epoch.append(loss.item())
#         # update weights
#         optimizer.step()
#         optimizer.zero_grad()
#     train_loss.append(np.mean(train_loss_epoch))

# if __name__ == '__main__':
#     torch.save(model.state_dict(), "model")

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CaliforniaDataset
from model import RegressionModel
from torch.utils.data import DataLoader
from trainer import Trainer
from transforms import normalize_data
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    model = RegressionModel()
    train_dataset = CaliforniaDataset("../data/train_data.csv")
    val_dataset = CaliforniaDataset("../data/val_data.csv")
    train_dataset, val_dataset = normalize_data(train_dataset, val_dataset)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["training"]["learning_rate"]
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"]
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"]
    )
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
    )
    trainer.fit(
        train_loader=train_dataloader, 
        val_loader=val_dataloader, 
        epochs=config["training"]["num_epochs"])
    trainer.print_score()
    torch.save(model.state_dict(), "../models/model")


if __name__ == '__main__':
    main()

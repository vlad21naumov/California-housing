import os

import hydra
import mlflow
import onnx
import torch
import torch.nn as nn
import torch.optim as optim
from dvc.api import DVCFileSystem
from mlflow.models import infer_signature
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset import CaliforniaDataset
from model import RegressionModel
from trainer import Trainer
from transforms import normalize_data


def log_params(config: DictConfig):
    mlflow.log_param("lr", config["trainer"]["learning_rate"])
    mlflow.log_param("epochs", config["trainer"]["num_epochs"])
    mlflow.log_param("batch_size", config["trainer"]["batch_size"])


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    fs = DVCFileSystem(".")
    for file_path in fs.find("/data", detail=False, dvc_only=True):
        fs.get_file(
            ".." + file_path,
            ".." + file_path,
        )

    model = RegressionModel()

    train_dataset = CaliforniaDataset(
        config["data_loading"]["train_dataset_path"], config
    )
    train_dataset, mean, std = normalize_data(train_dataset)

    val_dataset = CaliforniaDataset(config["data_loading"]["val_dataset_path"], config)

    val_dataset.data = (val_dataset.data - mean) / std

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["trainer"]["learning_rate"])
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["trainer"]["batch_size"]
    )
    val_dataloader = DataLoader(val_dataset, batch_size=config["trainer"]["batch_size"])

    mlflow.set_experiment(config["logging"]["experiment_name"])
    with mlflow.start_run(run_name=config["logging"]["run_name"]):
        mlflow.set_experiment_tag(
            config["logging"]["experiment_tag"], config["logging"]["version"]
        )
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
        )
        input, output = trainer.fit(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            epochs=config["trainer"]["num_epochs"],
        )

        log_params(config)
        model_signature = infer_signature(
            input.detach().cpu().numpy(), output.detach().cpu().numpy()
        )

        torch.onnx.export(model, input, config["model_params"]["saved_model_path"])
        mlflow.onnx.log_model(
            onnx.load(config["model_params"]["saved_model_path"]),
            config["logging"]["artifact_path"],
            registered_model_name=config["logging"]["model_name"],
            signature=model_signature,
        )

    state_dict = {"params": [mean, std]}

    print("Saving mean and std for inference...")
    torch.save(state_dict, config["model_params"]["saved_params_path"])
    print("Parametres are saved!")

    os.system("mlflow ui")


if __name__ == "__main__":
    main()

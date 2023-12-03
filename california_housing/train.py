import os

import hydra
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CaliforniaDataset
from dvc.api import DVCFileSystem
from model import RegressionModel
from omegaconf import DictConfig
from torch.utils.data import DataLoader
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
        # model_signature = infer_signature(
        #     input.detach().cpu().numpy(), output.detach().cpu().numpy()
        # )
        # print(input.shape)
        # onnx_model = torch.onnx.export(model, input, "model.onnx")
        # mlflow.onnx.log_model(
        #     onnx_model,
        #     "model",
        #     # registered_model_name=config["logging"]["model_name"],
        #     signature=model_signature,
        # )

    state_dict = {"model": model.state_dict(), "params": [mean, std]}

    print("Saving model and params...")
    torch.save(state_dict, config["model_params"]["saved_model_path"])
    print("Model and params are saved!")

    os.system("mlflow ui")


if __name__ == "__main__":
    main()

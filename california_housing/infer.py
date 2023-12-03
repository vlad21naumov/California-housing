import hydra
import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig
from onnx2torch.converter import convert

from dataset import CaliforniaDataset


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    logged_model_path = f'runs:/{config["logging"]["run_id"]}/model'
    onnx_model = mlflow.onnx.load_model(logged_model_path)
    model = convert(onnx_model)

    model_params = torch.load(config["model_params"]["saved_params_path"])

    mean, std = model_params["params"]

    test_dataset = CaliforniaDataset(config["data_loading"]["test_dataset_path"], config)
    test_dataset.data = (test_dataset.data - mean) / std

    model.eval()
    with torch.no_grad():
        predictions = model(test_dataset.data)
        predictions = predictions.detach().cpu().numpy()
        result = pd.DataFrame(
            test_dataset.data.numpy(), columns=config["data_loading"]["feature_columns"]
        )
        result[config["data_loading"]["target_column"]] = predictions

    result.to_csv(config["data_loading"]["result_path"])


if __name__ == "__main__":
    main()

import hydra
import pandas as pd
import torch
from dataset import CaliforniaDataset
from model import RegressionModel
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    model_params = torch.load(config["model_params"]["saved_model_path"])

    mean, std = model_params["params"]

    test_dataset = CaliforniaDataset(config["data_loading"]["test_dataset_path"], config)
    test_dataset.data = (test_dataset.data - mean) / std

    model = RegressionModel()

    model.load_state_dict(model_params["model"])

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

import numpy as np
import pandas as pd
import torch
from dataset import CaliforniaDataset
from model import RegressionModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transforms import normalize_data


def main():
    feature_names = [
        'MedInc',
        'HouseAge',
        'AveRooms',
        'AveBedrms',
        'Population',
        'AveOccup',
        'Latitude',
        'Longitude',
    ]
    train_dataset = CaliforniaDataset("../data/train_data.csv")
    test_dataset = CaliforniaDataset("../data/test_data.csv")
    _, test_dataset = normalize_data(train_dataset, test_dataset)
    model = RegressionModel()
    model.load_state_dict(torch.load("model"))
    model.eval()
    predictions = model(test_dataset.data)

    predictions = predictions.detach().numpy()
    result = pd.DataFrame(test_dataset.data, columns=feature_names)
    result["Price"] = predictions

    y_true = test_dataset.labels.reshape(-1, 1)
    print(f"MSE: {np.round(mean_squared_error(predictions, y_true), 2)}")
    print(f"MAE: {round(mean_absolute_error(predictions, y_true), 2)}")
    print(f"R2 score: {round(r2_score(predictions, y_true), 2)}")
    result.to_csv("predictions.csv")


if __name__ == '__main__':
    main()

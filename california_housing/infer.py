import numpy as np
import pandas as pd
import torch
from dataset import get_test_dataset
from model import RegressionModel
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


dataset = fetch_california_housing()
test_dataset = get_test_dataset()

model = RegressionModel()
model.load_state_dict(torch.load("model"))
model.eval()
predictions = model(test_dataset.data)

predictions = predictions.detach().numpy()
result = pd.DataFrame(test_dataset.data, columns=dataset.feature_names)
result["Price"] = predictions
result.to_csv("predictions.csv")

y_true = test_dataset.labels.reshape(-1, 1)
print(f"MSE: {np.round(mean_squared_error(predictions, y_true), 2)}")
print(f"MAE: {round(mean_absolute_error(predictions, y_true), 2)}")
print(f"R2 score: {round(r2_score(predictions, y_true), 2)}")

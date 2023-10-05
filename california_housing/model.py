import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()

        self.linear_1 = nn.Linear(8, 24)
        self.linear_2 = nn.Linear(24, 12)
        self.linear_3 = nn.Linear(12, 6)
        self.linear_4 = nn.Linear(6, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(self.activation(x))
        x = self.linear_3(self.activation(x))
        x = self.linear_4(self.activation(x))
        return x

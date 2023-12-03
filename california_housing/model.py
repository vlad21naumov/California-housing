import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        output = self.model(x)
        return output

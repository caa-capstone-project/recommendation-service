import torch.nn as nn
import torch.nn.functional as F

class Recommender(nn.Module):
    def __init__(self):
        super(Recommender, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9724, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 9724),
        )

    def forward(self, x):
        return self.model(x)

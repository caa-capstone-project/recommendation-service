import torch.nn as nn
import torch.nn.functional as F

class Recommender(nn.Module):
    def __init__(self):
        super(Recommender, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(9724, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 9724)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

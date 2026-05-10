import torch
import torch.nn as nn
from torchsummary import summary


class LatticeCNN(nn.Module):
    """
    A simple CNN for Lattice Gauge Theories.
    Uses circular padding to enforce periodic boundary conditions.
    """

    def __init__(self, L, D, channels):
        super(LatticeCNN, self).__init__()
        self.L = L
        self.D = D
        layers = []
        for chan_in, chan_out in zip(channels, channels[1:]):
            layers.append(
                nn.Conv2d(
                    chan_in, chan_out, kernel_size=3, padding=1, padding_mode="circular"
                )
            )
            layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        self.conv = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(self.L**self.D * channels[-1], 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze(-1)


if __name__ == "__main__":
    L = 5
    D = 2
    model = LatticeCNN(L, D, [1, 16, 32])
    summary(model, (1, L, L))

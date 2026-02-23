import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

        # Замораживаем веса
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x)
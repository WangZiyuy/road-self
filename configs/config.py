import torch

class Config:
    def __init__(self):
        self.circle_radius = torch.nn.Parameter(torch.tensor(20.0), requires_grad=True)
        self.neighborhood_radius = torch.nn.Parameter(torch.tensor(50.0), requires_grad=True)

config = Config()

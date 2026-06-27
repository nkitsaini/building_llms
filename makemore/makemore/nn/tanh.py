from .model import Model
import torch

class Tanh(Model):
    def __init__(self):
        self.training = True

    def __call__(self, x: torch.Tensor):
        y = ((2*x).exp())
        return (y - 1) / (y + 1)

    def parameters(self) -> list[torch.Tensor]:
        return []

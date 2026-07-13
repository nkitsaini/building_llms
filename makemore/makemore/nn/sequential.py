from .model import Model
import torch


class Sequential(Model):
    def __init__(self, layers: list[Model]):
        self.layers = layers

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)
        self.out = x
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return [p for l in self.layers for p in l.parameters()]

    def set_training(self, training: bool):
        for l in self.layers:
            l.training = training

from .model import Model
import torch

class BatchNorm(Model):
    def __init__(self, in_feat: int, eps: float = 1e-5, momentum: float = 0.1):
        self.gamma = torch.ones((1, in_feat))
        self.beta = torch.zeros((1, in_feat))

        self.running_mean = torch.zeros((1, in_feat))
        self.running_std = torch.ones((1, in_feat))

        self.eps = eps
        self.momentum = momentum
        self.training = True

    def __call__(self, x: torch.Tensor):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        if self.training:
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
            self.running_std = (1-self.momentum)*self.running_std + self.momentum*std

        result = (x - mean)/((std**2 + self.eps)**0.5)
        with torch.no_grad():
            return result*self.gamma + self.beta

    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]

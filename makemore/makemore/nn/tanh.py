from .model import Model
import torch


class Tanh(Model):
    def __init__(self):
        self.training = True
        self.x = None

    def __call__(self, x: torch.Tensor):
        self.x = x
        y = (2 * x).exp()
        self.out = (y - 1) / (y + 1)
        return self.out

    def backprop(self, out_grad: torch.Tensor) -> torch.Tensor:
        assert self.x is not None
        with torch.no_grad():
            x_exp = (2*self.x).exp()
            return (4*x_exp)/((x_exp +1)**2) * out_grad
            # return xgrad = (1 - self.out**2)  * out_grad

    def parameters(self) -> list[torch.Tensor]:
        return []

from .model import Model
import torch


class BatchNorm(Model):
    def __init__(self, in_feat: int, eps: float = 1e-5, momentum: float = 0.1):
        self.gamma = torch.ones((in_feat,))
        self.beta = torch.zeros((in_feat,))

        self.running_mean = torch.zeros((in_feat,))
        self.running_var = torch.ones((in_feat,))

        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.x = None
        self.result = None

    def __call__(self, x: torch.Tensor):
        self.x = x
        if self.training:
            # NB: dim=0 means we want to calculate per "neuron", not per sample
            mean = x.mean(0, keepdim=True)
            var = x.var(0, keepdim=True)
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        result = (x - mean) / ((var + self.eps) ** 0.5)
        self.result = result
        self.out = result * self.gamma + self.beta
        return self.out

    # (in_feat, )
    def backprop(self, out_grad: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError()
        # assert self.result is not None
        # assert self.x is not None
        # with torch.no_grad():
        #     self.init_grads()
        #     self.beta.grad += out_grad # type: ignore
        #     self.gamma.grad += self.result.sum(0) * out_grad # type: ignore
        #     ...

    def init_grad(self):
        raise NotImplementedError()

    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]

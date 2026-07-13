import abc
import torch


class Model(abc.ABC):
    training: bool
    out: torch.Tensor | None

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def parameters(self) -> list[torch.Tensor]: ...

    # @abc.abstractmethod
    # def backprop(self, out_grad: torch.Tensor) -> torch.Tensor: ...

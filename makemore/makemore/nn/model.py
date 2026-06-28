import abc
import torch

class Model(abc.ABC):
    training: bool

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def parameters(self) -> list[torch.Tensor]:
        ...

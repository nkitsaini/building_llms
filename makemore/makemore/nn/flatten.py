from .model import Model
import torch


class FlattenConsecutive(Model):
    def __init__(self, n: int):
        self.n = n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, vocab_size, embed_size = x.shape
        assert vocab_size % self.n == 0, (
            f"Can't flatten due to {vocab_size=} not being multiple of {self.n=}"
        )
        x = x.view(batch_size, vocab_size // self.n, embed_size * self.n)
        if x.size(1) == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return []

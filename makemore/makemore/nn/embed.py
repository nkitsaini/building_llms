from IPython import embed
from .model import Model
import torch


class Embedding(Model):
    def __init__(
        self,
        num_embedding: int,
        embedding_dim: int,
        generator: torch.Generator | None = None,
        device: torch.Device | None = None,
    ):
        self.embeddings = torch.randn(
            (num_embedding, embedding_dim), generator=generator, device=device
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.out = self.embeddings[x]
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return [self.embeddings]

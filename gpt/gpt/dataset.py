from pathlib import Path
from typing import Literal
import torch


class TinyShakespeareDataset:
    def __init__(self, content: str, device: torch.Device = None):
        self.content = content
        self.chars = sorted(list(set(content)))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = self.chars
        self.tensor = torch.tensor(self.encode(self.content), dtype=torch.long)
        split_idx = int(0.9 * len(self.tensor))
        self.train_data = self.tensor[:split_idx]
        self.val_data = self.tensor[split_idx:]
        self.device = device

    def encode(self, content: str) -> list[int]:
        return [self.stoi[c] for c in content]

    def get_batch(self, dataset: Literal['train', 'val'] = 'train', batch_size: int = 4, block_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if dataset == 'train' else self.val_data
        idx = torch.randint(len(data)-block_size, size=(batch_size,))
        x = torch.stack([data[i:i+block_size] for i in idx]).to(self.device)
        y = torch.stack([data[i+1:i+block_size+1] for i in idx]).to(self.device)
        return x, y

    def decode(self, encoded: list[int]) -> str:
        return "".join(self.itos[i] for i in encoded)

    @property
    def vocab_size(self):
        return len(self.chars)

    @classmethod
    def from_file(cls, path: Path | str, device: torch.Device = None) -> "TinyShakespeareDataset":
        return cls(Path(path).read_text(), device=device)


def load_dataset(device: torch.Device = None) -> TinyShakespeareDataset:
    path = Path(__file__).parent / "dataset/tinyshakespeare.txt"
    return TinyShakespeareDataset.from_file(path, device)

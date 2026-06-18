import string

import torch

START_TOKEN = "."
END_TOKEN = "."

CHARS = [START_TOKEN, *string.ascii_lowercase]


def create_count_tensor(words: list[str]) -> torch.Tensor:
    counts = torch.zeros(len(CHARS), len(CHARS), dtype=torch.int)
    for word in words:
        chars = [START_TOKEN, *word, END_TOKEN]
        for c1, c2 in zip(chars, chars[1:]):
            counts[char_to_int(c1), char_to_int(c2)] += 1
    return counts

def create_training_data(words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for word in words:
        chars = [START_TOKEN, *word, END_TOKEN]
        for c1, c2 in zip(chars, chars[1:]):
            xs.append(char_to_int(c1))
            ys.append(char_to_int(c2))
    return torch.tensor(xs), torch.tensor(ys)

def char_to_int(char: str) -> int:
    if char == START_TOKEN:
        return 0
    assert len(char) == 1
    char = char.lower()
    return ord(char) - ord("a") + 1

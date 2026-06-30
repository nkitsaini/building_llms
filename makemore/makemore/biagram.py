import string

from pathlib import Path
import torch

START_TOKEN = "."
END_TOKEN = "."

CHARS = [START_TOKEN, *string.ascii_lowercase]


def read_words() -> list[str]:
    return (Path(__file__).parent.parent / "names.txt").read_text().splitlines()


def create_count_tensor(words: list[str]) -> torch.Tensor:
    counts = torch.zeros(len(CHARS), len(CHARS), dtype=torch.int)
    for word in words:
        chars = [START_TOKEN, *word, END_TOKEN]
        for c1, c2 in zip(chars, chars[1:]):
            counts[char_to_int(c1), char_to_int(c2)] += 1
    return counts


def create_training_data(
    words: list[str], block_size: int = 1, debug: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for word in words:
        padding = [START_TOKEN] * block_size
        chars = [*padding, *word, END_TOKEN]
        if debug:
            print(word)
        for i in range(len(word) + 1):
            x_char = chars[i : i + block_size]
            y_char = chars[i + block_size]
            if debug:
                print(f"{''.join(x_char)} ---> {y_char}")
            x = [char_to_int(c) for c in x_char]
            xs.append(x)
            ys.append(char_to_int(y_char))
    xs, ys = torch.tensor(xs), torch.tensor(ys)
    if xs.size(1) == 1:
        xs = xs.squeeze()
    return xs, ys


def char_to_int(char: str) -> int:
    if char == START_TOKEN:
        return 0
    assert len(char) == 1
    char = char.lower()
    return ord(char) - ord("a") + 1

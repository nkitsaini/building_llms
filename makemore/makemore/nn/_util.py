from torch import Tensor


def grad_add(existing: Tensor | None, new: Tensor) -> Tensor:
    if existing is None:
        return new
    return existing + new

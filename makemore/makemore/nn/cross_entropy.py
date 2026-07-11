from .model import Model
import torch
import torch.nn
import torch.nn.functional as F


def cross_entropy(x: torch.Tensor, ypred: torch.Tensor) -> torch.Tensor:
    """
    logits.shape = [*ypred.shape, N]
    where N is the number of total classes
    """
    x = x - x.max()
    xexp = x.exp()
    probs = xexp / xexp.sum(-1, keepdim=True)
    prob_logs = probs.log()
    pred_prob_logs = prob_logs[torch.arange(0, len(ypred)), ypred]
    loss = pred_prob_logs.mean().neg()
    return loss


def backprop_cross_entropy(out_grad: float = 1.):
    # TODO
    raise NotImplementedError()
    # dpred_prob_logs = torch.ones_like() -out_grad * 1/torch.
    ...

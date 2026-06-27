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
    probs = xexp/xexp.sum(-1, keepdim=True)
    pred_probs = probs[torch.arange(0, len(ypred)), ypred]
    pred_prob_logs = pred_probs.log()
    loss = pred_prob_logs.mean().neg()
    return loss



def test_cross_entropy():
    ...

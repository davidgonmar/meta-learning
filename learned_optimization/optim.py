from typing import Sequence
import torch
from torch import nn
from learned_optimization.models import TinyRNN


def _apply_update(params, upd, shapes):
    new_params = []
    idx = 0
    for p, shape in zip(params, shapes):
        size = p.numel()
        upd_slice = upd[idx : idx + size].reshape(shape)
        new_params.append(p + upd_slice)
        idx += size
    return new_params


def _flatten_params(params: Sequence[torch.Tensor]) -> torch.Tensor:
    flat_params = [p.view(-1) for p in params if p is not None]
    return torch.cat(flat_params), [p.shape for p in params if p is not None]


def _flatten_params_and_grads(params, grads):
    flat_p, flat_g, shapes = [], [], []
    for p, g in zip(params, grads):
        flat_p.append(p.view(-1))
        flat_g.append(g.view(-1))
        shapes.append(p.shape)
    return torch.cat(flat_p), torch.cat(flat_g), shapes


def preprocess_grad(grad: torch.Tensor, p: float = 10.0) -> torch.Tensor:
    p_t = torch.as_tensor(p, dtype=grad.dtype, device=grad.device)
    eps = torch.finfo(grad.dtype).tiny
    thresh = torch.exp(-p_t).clamp(min=eps)
    abs_g = grad.abs()
    large = abs_g >= thresh
    mag_large = torch.log(abs_g + eps) / p_t
    sign_large = torch.sign(grad)
    mag_small = torch.full_like(grad, -1.0)
    sign_small = grad * torch.exp(p_t)
    mag = torch.where(large, mag_large, mag_small)
    sign = torch.where(large, sign_large, sign_small)

    return torch.stack([mag, sign], dim=-1)


class LearnedOptimizer(nn.Module):
    def __init__(self, hidden: int = 6):
        super().__init__()
        self.rnn = TinyRNN(hidden)

    def init_state(
        self, params: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(p for p in params if p is not None).device
        hidden_size = self.rnn.lstm.hidden_size
        flat_p, _ = _flatten_params(params)
        h0 = torch.zeros(
            self.rnn.num_layers, flat_p.numel(), hidden_size, device=device
        )
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    def forward(
        self,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[list[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        flat_p, flat_g, shapes = _flatten_params_and_grads(params, grads)
        flat_g = preprocess_grad(flat_g)
        inp = flat_g.unsqueeze(1)
        upd_seq, new_state = self.rnn(inp, state)
        upd = upd_seq.squeeze(1)
        updated_params = _apply_update(params, upd[:, 0] * 0.1, shapes)
        return updated_params, new_state

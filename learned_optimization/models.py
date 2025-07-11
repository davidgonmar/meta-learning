import torch
from torch import nn


class TinyRNN(nn.Module):
    def __init__(self, hidden: int = 10):
        super().__init__()
        self.num_layers = 1
        self.hidden_size = hidden
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 2)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | tuple
    ) -> tuple[torch.Tensor, tuple]:
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        if not isinstance(state, tuple):
            h0 = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=device,
                dtype=dtype,
            )
            h0[0] = state
            c0 = torch.zeros_like(h0)
        else:
            h0, c0 = state

        out, (hn, cn) = self.lstm(x, (h0, c0))
        upd = self.head(out[:, -1, :]).unsqueeze(1)
        return upd, (hn, cn)

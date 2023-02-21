from torch import nn
from functools import partial
import torch.nn.functional as F


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


def MLP(num_patches, dim_ff):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
    return nn.Sequential(
        FeedForward(num_patches, dense=chan_first),
        FeedForward(dim_ff, dense=chan_last)
    )


class MLP_full(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
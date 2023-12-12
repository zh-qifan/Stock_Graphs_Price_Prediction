from typing import Optional, Tuple, Union
from torch import Tensor
from torch import concat
import torch.nn as nn
from torch_geometric.nn import GATConv, to_hetero_with_bases
from .lstm import LSTMModule
from .decoder import Decoder

class GATv3(GATConv):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            fill_value=fill_value,
            bias=bias,
            **kwargs
            )
        
    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        m = alpha.unsqueeze(-1) * x_j
        return m.reshape(m.shape[0], -1)

class GATv3Module(nn.Module):
    def __init__(
        self, 
        input_size, 
        output_size, 
        num_heads,
        dropout=0.5
    ):
        super(GATv3Module, self).__init__()
        self.conv1 = GATv3(in_channels=input_size, out_channels=output_size, heads=num_heads, dropout=dropout, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class LSTMGATBasis(nn.Module):
    
    def __init__(
        self, 
        input_size, 
        sequence_length, 
        lstm_hidden_size, 
        metadata,
        num_bases,
        output_size=1, 
        num_heads=1, 
        dropout=0.5
    ):
        super(LSTMGATBasis, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.metadata = metadata
        self.lstm = LSTMModule(input_size, sequence_length, lstm_hidden_size)
        self.gat = GATv3Module(lstm_hidden_size, lstm_hidden_size, num_heads=num_heads, dropout=dropout)
        self.gat = to_hetero_with_bases(self.gat, self.metadata, num_bases=num_bases, in_channels={"x": lstm_hidden_size}, debug=False)
        self.gat_corr = GATv3Module(lstm_hidden_size, lstm_hidden_size, num_heads=num_heads, dropout=dropout)
        self.decoder = Decoder(lstm_hidden_size * (2 * num_heads + 1), output_size)

    def forward(self, x, edge_index):
        edge_index_hetero, edge_index = edge_index
        t_x = self.lstm(x)
        t_x = t_x[:, -1, :]
        r_x = self.gat({"stock" : t_x}, edge_index_hetero)
        r_x_corr = self.gat_corr(t_x, edge_index)
        x = self.decoder(concat((t_x, r_x["stock"], r_x_corr), dim=-1))
        return x

import torch.nn as nn
from torch import concat, arange
from torch_geometric.nn import GATConv, to_hetero
from .lstm import LSTMModule
from .decoder import Decoder
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
)

class GAT(nn.Module):
    def __init__(
        self, 
        input_size, 
        output_size, 
        num_heads,
        dropout=0.5
    ):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels=input_size, out_channels=output_size, heads=num_heads, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

class LSTMGAT(nn.Module):
    def __init__(
        self, 
        input_size, 
        sequence_length, 
        lstm_hidden_size, 
        metadata,
        output_size=1, 
        num_heads=1, 
        dropout=0.5
    ):
        super(LSTMGAT, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.metadata = metadata
        self.lstm = LSTMModule(input_size, sequence_length, lstm_hidden_size)
        self.gat = GAT(lstm_hidden_size, lstm_hidden_size, num_heads, dropout)
        self.gat = to_hetero(self.gat, self.metadata)
        self.gat_corr = GAT(lstm_hidden_size, lstm_hidden_size, num_heads, dropout)
        self.decoder = Decoder(lstm_hidden_size * (2 * num_heads + 1), output_size)
    
    def forward(self, x, edge_index):
        edge_index_hetero, edge_index = edge_index
        t_x = self.lstm(x)
        t_x = t_x[:, -1, :]
        r_x = self.gat({"stock" : t_x}, edge_index_hetero)
        r_x_corr = self.gat_corr(t_x, edge_index)
        x = self.decoder(concat((t_x, r_x["stock"], r_x_corr), dim=-1))
        return x

    def att_mat(self, model, x, edge_index):
        H, C = model.heads, model.out_channels
        x_src = model.lin_src(x).view(-1, H, C)
        x_dst = model.lin_dst(x).view(-1, H, C)
        x = (x_src, x_dst)
        
        alpha_src = (x_src * model.att_src).sum(dim=-1)
        alpha_dst = (x_dst * model.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)
        
        if model.add_self_loops:
            num_nodes = x_src.size(0)
            if x_dst is not None:
                num_nodes = min(num_nodes, x_dst.size(0))
            edge_index, _ = remove_self_loops(
                edge_index, None)
            edge_index, _ = add_self_loops(
                edge_index, None, fill_value=model.fill_value,
                num_nodes=num_nodes)
        
        alpha = model.edge_updater(edge_index, alpha=alpha, edge_attr=None)
        
        return alpha, edge_index

    def get_att_mat(self, x, edge_index):
        t_x = self.lstm(x)
        t_x = t_x[:, -1, :]
        keys = list(edge_index.keys())
        ori_keys = {"__".join(list(i)) : i for i in keys}
        alphas = {}
        self_loop_edge_indexes = {}
        for k, m in self.gat.get_submodule("conv1").items():
            alpha, self_loop_edge_index = self.att_mat(m, t_x, edge_index[ori_keys[k]])
            alpha = alpha.cpu().detach().numpy()
            self_loop_edge_index = self_loop_edge_index.cpu().detach().numpy()
            alphas[k] = alpha
            self_loop_edge_indexes[k] = self_loop_edge_index
        return alphas, self_loop_edge_indexes
    
class LSTMGATAug(nn.Module):
    def __init__(
        self, 
        input_size, 
        sequence_length, 
        lstm_hidden_size, 
        metadata,
        output_size=1, 
        num_heads=1, 
        dropout=0.5
    ):
        super(LSTMGATAug, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.metadata = metadata
        self.lstm = LSTMModule(input_size, sequence_length, lstm_hidden_size)
        self.gat = GAT(lstm_hidden_size, lstm_hidden_size, num_heads, dropout)
        self.gat = to_hetero(self.gat, self.metadata)
        self.gat_corr = GAT(lstm_hidden_size, lstm_hidden_size, num_heads, dropout)
        self.decoder = Decoder(lstm_hidden_size * (2 * num_heads + 1), output_size)

    def forward(self, x, edge_index):
        edge_index_hetero, edge_index = edge_index
        x = self.lstm(x)
        t_x = x[:, -1, :]
        x = x.reshape(x.shape[1] * x.shape[0], x.shape[2])
        r_x = self.gat({"stock" : x}, edge_index_hetero)
        r_x = r_x["stock"]
        idx = arange(self.sequence_length - 1, r_x.shape[0], self.sequence_length).to(r_x.device)
        r_x = r_x[idx]
        r_x_corr = self.gat_corr(t_x, edge_index)
        x = self.decoder(concat((t_x, r_x, r_x_corr), dim=-1))
        return x



        
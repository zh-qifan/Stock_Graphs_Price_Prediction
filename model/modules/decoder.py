import torch.nn as nn

class Decoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        out_channels,
    ):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(emb_dim, out_channels)
    
    def forward(self, x):
        x = self.linear(x)
        return x
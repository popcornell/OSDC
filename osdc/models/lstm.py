from torch import nn
from asteroid.masknn import norms

class LSTMDense(nn.Module):

    def __init__(self, input_size, n_outs=5, hidden_sizes=(512, 1024, 512, 256), bidirectional=False):
        super(LSTMDense, self).__init__()
        self.norm = norms.get("gLN")(input_size)
        self.lstm = nn.LSTM(input_size, hidden_sizes[0], bidirectional=bidirectional)

        out_feats = hidden_sizes[0] if bidirectional == False else hidden_sizes[0]*2

        self.denses = nn.Sequential(nn.Linear(out_feats, hidden_sizes[1]), nn.ReLU(),
                                    nn.Linear(hidden_sizes[1], hidden_sizes[2]), nn.ReLU(),
                                    nn.Linear(hidden_sizes[2], hidden_sizes[3]), nn.ReLU())

        self.out = nn.Sequential(nn.Linear(hidden_sizes[-1], n_outs), nn.Softmax())

    def forward(self, feats):
        feats = self.norm(feats)
        out, _ = self.lstm(feats.transpose(1, -1))
        out = self.denses(out)
        return self.out(out).transpose(1,-1)


if __name__ == "__main__":
    import torch
    a = torch.rand((2, 64, 1000))
    lstm = LSTMDense(64)
    lstm(a)
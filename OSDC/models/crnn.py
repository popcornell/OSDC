import torch
from torch import nn
from asteroid.masknn import norms

class C2DRNN(nn.Module): # modified one with skip connection and 1-D convs

    def __init__(self, in_size, n_outs=3,  repeats=2, blocks=2, channels=(64, 32, 128, 64),
                 hidden_size=40, bidirectional=False, ksz=(3, 3), dropout=0, activation=nn.ReLU()):
        super(C2DRNN, self).__init__()
        self.in_size = in_size


        assert len(channels) == repeats*blocks
        self.norm = norms.get("gLN")(in_size)
        net = []
        for i in range(repeats):
            for j in range(blocks):

                if i == 0 and j == 0:
                    conv_in = 1
                else:
                    conv_in = channels[i*2+j-1]

                net.extend([nn.Conv2d(conv_in, channels[i*2+j], ksz, 1), activation])

            net.append(nn.MaxPool2d(ksz))

        net.append(nn.Dropout(dropout))

        self.feats = nn.Sequential(*net)
        self.lstm = nn.LSTM(64*5, hidden_size, bidirectional=bidirectional)
        feats_in = hidden_size if not bidirectional else hidden_size*2
        self.max1d = nn.MaxPool1d(2, stride=2)
        self.out = nn.Sequential(nn.Linear(1040, n_outs), nn.Softmax(-1))

    def forward(self, x):
        x = self.norm(x)
        x = x.transpose(1, -1).unsqueeze(1)
        x =  self.feats(x)
        B, C, D, F = x.size()
        x = x.transpose(1, 2).reshape(B, D, C*F)
        x, _ = self.lstm(x)
        x = self.max1d(x.transpose(1, -1))
        B, C, F = x.size()
        x = self.out(x.reshape(B, C*F))

        return x

if __name__ == "__main__":

    b = torch.rand((2, 64, 498))

    a = C2DRNN(64, 5)
    print(a(b).shape)

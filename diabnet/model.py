import torch
from torch import nn

class LocallyConnected(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, bias=False):
        super(LocallyConnected, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, output_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, output_size))
        else:
            self.register_parameter('bias', None)


    def forward(self, x):
        out = (x.unsqueeze(1) * self.weight).sum(2)
        if self.bias is not None:
            out += self.bias
        out = out.squeeze(1)
        return out

class Model(nn.Module):
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_hidden_3, dropout_p0, dropout_p1, dropout_p2, dropout_p3):
        super(Model, self).__init__()
        self.lc = LocallyConnected(2, 1, n_feat, bias=False)
        self.l1 = nn.Linear(n_feat, n_hidden_1)
        # self.l2 = nn.Linear(n_hidden_1, n_hidden_2)
        # self.l3 = nn.Linear(n_hidden_2, n_hidden_1)
        self.exit = nn.Linear(n_hidden_1, 1)
        p0 = dropout_p0
        self.drop0 = nn.Dropout(p=p0)
        p1 = dropout_p1
        self.drop1 = nn.Dropout(p=p1)
        # p2 = dropout_p2
        # self.drop2 = nn.Dropout(p=p2)
        # p3 = dropout_p3
        # self.drop3 = nn.Dropout(p=p3)
        self.act = nn.Softplus()
        self.bn = nn.BatchNorm1d(n_hidden_1)


    def forward(self, x):
        out = self.lc(x)
        # out = self.bn(out)
        out = self.drop0(out)
        # out = self.act(out)
        out = self.l1(out)
        out = self.bn(out)
        out = self.act(out)
        
        out = self.drop1(out)
        # out = self.l2(out)
        # out = self.act(out)
        # out = self.drop2(out)
        
        # out = self.l3(out)
        # out = self.drop3(out)
        # out = self.act(out)

        # inp = self.act(out)
        # inp = self.l2(inp)
        # inp = self.act(inp)
        # inp = self.drop2(inp)
        # inp = self.l3(inp)
        # out = out + inp
        # out = self.drop1(out)
        # out = self.act(out)

        out = self.exit(out)

        return out

def load(filename: str):
    net = torch.load(filename, map_location='cpu')
    # net = net.to(device)
    net.eval() #! comment this line when using mc-dropout
    return net

if __name__ == "__main__":
    f = torch.zeros(2,3,4)

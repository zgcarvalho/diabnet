# import typing
import torch
from torch import nn


class LocallyConnected(nn.Module):
    """LocallyConnected Layer generates a single value representation for each column from the
    input matrix.

    In columns that contain SNP information the first line have the number of
    dominant alleles and the second the number of recessive alleles. More information about
    this and other features in 'data.py'.

    Default option are not use bias if no activation is used. Without activation function,
    the values encoded are linear, so an hetero SNP will have the mean value between both homo
    SNPs. With activation funtion, there is no linearity and the hetero SNP could have any value
    between the values of both homo SNPs but not greater or smaller than both of them. This
    characteristic resembles genetic effects like dominance. However may be necessary more data
    to acquiry a good parametrization because some SNPs are very rare and homo recessive individuals
    are difficult to observed.

    Args:
        output_size (int): number of values after lc layer. This number must be the same as the
        number of columns in the input matrix (n_feat)
        bias (bool): use bias (default: False)
        activation (bool): use activation function tanh (default: False)
    """
    def __init__(self, output_size: int, bias=False, activation=''):
        super(LocallyConnected, self).__init__()
        in_channels = 2 # number of inputs for each lc neuron
        out_channels = 1 # number of outpus for each lc neuron

        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, output_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, output_size))
        else:
            self.register_parameter('bias', None)

        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()



    def forward(self, x):
        out = (x.unsqueeze(1) * self.weight).sum(2)
        if self.bias is not None:
            out += self.bias
        out = out.squeeze(1)
        # if self.activation == 'tanh':
            # out = torch.tanh(out)
            # out = torch.sigmoid(out)
            # out = nn.functional.relu(out)
        out = self.act(out)
        return out


class Model(nn.Module):
    '''
    Args:
        n_feat (int):
        n_hidden (int):
        dropout_p (float):
        lc_layer (str):
        soft_label_baseline (float):
        soft_label_topline (float):

    '''
    def __init__(self,
                 n_feat: int,
                 n_hidden: int,
                 dropout_p: float,
                 lc_layer: str,
                 use_correction: bool,
                 soft_label_baseline: float,
                 soft_label_topline: float,
                 soft_label_baseline_slope: float,
                 age_idx: int):

        super(Model, self).__init__()
        if lc_layer == 'lc1':
            self.lc = LocallyConnected(n_feat, bias=False, activation='identity')
        elif lc_layer == 'lc2':
            self.lc = LocallyConnected(n_feat, bias=True, activation='tanh')
        elif lc_layer == 'lc3':
            self.lc = LocallyConnected(n_feat, bias=False, activation='tanh')
        elif lc_layer == 'lc4':
            self.lc = LocallyConnected(n_feat, bias=True, activation='sigmoid')
        elif lc_layer == 'lc5':
            self.lc = LocallyConnected(n_feat, bias=False, activation='sigmoid')
        elif lc_layer == 'lc6':
            self.lc = LocallyConnected(n_feat, bias=True, activation='gelu')
        elif lc_layer == 'lc7':
            self.lc = LocallyConnected(n_feat, bias=False, activation='gelu')

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(n_feat, n_hidden, bias=False)
        self.bn = nn.BatchNorm1d(n_hidden)
        #self.act = nn.Softplus()
        #self.act = nn.ReLU()
        self.act = nn.GELU()
        #self.act = nn.ELU()
        self.exit = nn.Linear(n_hidden, 1)

        self.use_correction = use_correction
        self.soft_label_baseline = soft_label_baseline
        self.soft_label_topline = soft_label_topline
        self.soft_label_baseline_slope = soft_label_baseline_slope
        self.age_idx = age_idx 
        # age_idx = training_set.dataset.feat_names.index('AGE')


    def forward(self, x):
        out = self.lc(x)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.exit(out)
        return out

    def apply(self, x):
        y = self.forward(x)
        return self.sigmoid(y, self.get_ages(x), with_correction=self.use_correction)

    def sigmoid(self, y, ages, with_correction=False):
        y = torch.sigmoid(y)
        if with_correction:
            if self.soft_label_baseline_slope == 0.0:
                base = torch.ones(1).to(y.device) * self.soft_label_baseline
                top = torch.ones(1).to(y.device) * self.soft_label_topline
                return torch.minimum(torch.maximum((y - base)/(top - base), torch.zeros(1).to(y.device)), torch.ones(1).to(y.device))
            else:
                ages = torch.minimum(torch.maximum(ages, torch.tensor([0.4]).to(y.device)), torch.tensor([1.6]).to(y.device))
                base = torch.maximum(self.soft_label_baseline + self.soft_label_baseline_slope * (ages - 0.4)/(1.6 - 0.4), torch.zeros(1).to(y.device))
                top = torch.ones(1).to(y.device) * self.soft_label_topline
                return torch.minimum(torch.maximum((y - base)/(top - base), torch.zeros(1).to(y.device)), torch.ones(1).to(y.device))
        return y

    def get_ages(self, x):
        return x[:,0:1,self.age_idx]


def load(filename: str):
    net = torch.load(filename, map_location='cpu')
    # net = net.to(device)
    net.eval() #! comment this line when using mc-dropout
    return net

if __name__ == "__main__":
    f = torch.zeros(2,3,4)

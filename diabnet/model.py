import torch
from torch import nn

__all__ = ["Model", "load"]


class LocallyConnected(nn.Module):
    """LocallyConnected Layer generates a single value representation for each
    column from the input matrix.

    In columns that contain SNP information the first line have the number of
    dominant alleles and the second the number of recessive alleles. More
    information about this and other features in 'data.py'.

    Default option are not use bias if no activation is used. Without
    activation function, the values encoded are linear, so an hetero SNP will
    have the mean value between both homo SNPs. With activation funtion, there
    is no linearity and the hetero SNP could have any value between the values
    of both homo SNPs but not greater or smaller than both of them. This
    characteristic resembles genetic effects like dominance. However may be
    necessary more data to acquiry a good parametrization because some SNPs
    are very rare and homo recessive individuals are difficult to observed.
    """

    def __init__(
        self,
        output_size: int,
        in_channels: int = 2,
        out_channels: int = 1,
        bias: bool = False,
        activation: str = "identity",
    ):
        """
        Parameters
        ----------
        output_size : int
            Number of values after lc layer. This number must be the same as the
            number of columns in the input matrix (n_feat).
        in_channels : int, optional
            Number of inputs for each lc neuron, by default 2.
        out_channels : int, optional
            Number of outputs for each lc neuron, by default 1.
        bias : bool, optional
            Whether to use bias, by default False.
        activation : str, optional
            A activation function to be used, by default `identity`.

        Raises
        ------
        ValueError
            `activation` must be `tanh`, `sigmoid`, `gelu` or `identity`.
        """
        super(LocallyConnected, self).__init__()

        # Define weights
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size)
        )

        # Define bias
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, output_size))
        else:
            self.register_parameter("bias", None)

        # Define activation function
        if activation.lower() == "tanh":
            self.act = nn.Tanh()
        elif activation.lower() == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "identity":
            self.act = nn.Identity()
        else:
            raise ValueError(
                "`activation` must be `tanh`, `sigmoid`, `gelu` or `identity`."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (x.unsqueeze(1) * self.weight).sum(2)
        if self.bias is not None:
            out += self.bias
        out = out.squeeze(1)
        out = self.act(out)
        return out


class Model(nn.Module):
    """
    DiabNet Neural Network, composed of three layers.
    Input Layer: Locally Connected Layer
    Hidden Layer: Fully Connected Layer
    Output Layer: Linear Layer
    """

    def __init__(
        self,
        n_feat: int,
        n_hidden: int,
        dropout_p: float,
        lc_layer: str,
        use_correction: bool,
        soft_label_baseline: float,
        soft_label_topline: float,
        soft_label_baseline_slope: float,
        age_idx: int,
    ):
        """
        Parameters
        ----------
        n_feat : int
            Number of features in the dataset.
        n_hidden : int
            Number of neurons in the hidden layer.
        dropout_p : float
            Probability of an element to be zeroed in Dropout.
        lc_layer : str
            Locally Connected Layer to be used in the Model. Accepted values
            are: `lc1`, `lc2`, `lc3`, `lc4`, `lc5`, `lc6` and `lc7`.
        soft_label_baseline : float
            Soft label baseline
        soft_label_topline : float
            Soft label topline
        soft_label_baseline_slope : float
            Soft label slope
        age_idx : int
            Index of age column in the dataset
        """
        super(Model, self).__init__()

        # Input Layer
        if lc_layer == "lc1":
            self.lc = LocallyConnected(n_feat, bias=False, activation="identity")
        elif lc_layer == "lc2":
            self.lc = LocallyConnected(n_feat, bias=True, activation="tanh")
        elif lc_layer == "lc3":
            self.lc = LocallyConnected(n_feat, bias=False, activation="tanh")
        elif lc_layer == "lc4":
            self.lc = LocallyConnected(n_feat, bias=True, activation="sigmoid")
        elif lc_layer == "lc5":
            self.lc = LocallyConnected(n_feat, bias=False, activation="sigmoid")
        elif lc_layer == "lc6":
            self.lc = LocallyConnected(n_feat, bias=True, activation="gelu")
        elif lc_layer == "lc7":
            self.lc = LocallyConnected(n_feat, bias=False, activation="gelu")

        # Hidden Layer
        # Define dropout
        self.dropout = nn.Dropout(p=dropout_p)
        # Define fully connected with batch normalization
        self.fc = nn.Linear(n_feat, n_hidden, bias=False)
        self.bn = nn.BatchNorm1d(n_hidden)
        # Define activation function
        # [Options: GELU, Softplus, ReLU, ELU]
        self.act = nn.GELU()

        # Output Layer
        self.exit = nn.Linear(n_hidden, 1)
        # Define soft label correction
        self.use_correction = use_correction
        self.soft_label_baseline = soft_label_baseline
        self.soft_label_topline = soft_label_topline
        self.soft_label_baseline_slope = soft_label_baseline_slope
        # Define age index
        self.age_idx = age_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lc(x)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.exit(out)
        return out

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        ages = self.get_ages(x)
        return self.sigmoid(y, ages, with_correction=self.use_correction)

    def sigmoid(
        self, y: torch.Tensor, ages: torch.Tensor, with_correction: bool = False
    ) -> torch.Tensor:
        # Apply sigmoid to elements of y
        y = torch.sigmoid(y)

        # Apply soft label correction
        if with_correction:
            if self.soft_label_baseline_slope == 0.0:
                base = torch.ones(1).to(y.device) * self.soft_label_baseline
                top = torch.ones(1).to(y.device) * self.soft_label_topline
                return torch.minimum(
                    torch.maximum(
                        (y - base) / (top - base), torch.zeros(1).to(y.device)
                    ),
                    torch.ones(1).to(y.device),
                )
            else:
                ages = torch.minimum(
                    torch.maximum(ages, torch.tensor([0.4]).to(y.device)),
                    torch.tensor([1.6]).to(y.device),
                )
                base = torch.maximum(
                    self.soft_label_baseline
                    + self.soft_label_baseline_slope * (ages - 0.4) / (1.6 - 0.4),
                    torch.zeros(1).to(y.device),
                )
                top = torch.ones(1).to(y.device) * self.soft_label_topline
                return torch.minimum(
                    torch.maximum(
                        (y - base) / (top - base), torch.zeros(1).to(y.device)
                    ),
                    torch.ones(1).to(y.device),
                )
        return y

    def get_ages(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0:1, self.age_idx]


def load(filename: str) -> Model:
    """Load parameters from previously trained models.

    Parameters
    ----------
    filename : str
        A path to a .pth file with the model parameters.

    Returns
    -------
    Model
        A previously trained model.
    """
    # Load previously trained paramaters to Model
    net = torch.load(filename, map_location="cpu")

    # ! Comment this line when using mc-dropout
    # This method switch Model to `eval` mode.
    # NOTE: Dropout an BatchNorm behave differently during training
    # and evaluation
    net.eval()

    return net

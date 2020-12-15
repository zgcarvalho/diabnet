import torch
from torch import nn


class CalibrationModel(nn.Module):
    def __init__(self, model):
        super(CalibrationModel, self).__init__()
        model.eval()
        self.model = model
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x):
        h = self.alpha * self.model(x) + self.beta
        return h


def ece_mce(preds, targets, bins=10):
    lower_bound = torch.arange(0.0, 1.0, 1.0 / bins)
    upper_bound = lower_bound + 1.0 / bins

    ece = torch.zeros(1, device=preds.device)
    mce = torch.zeros(1, device=preds.device)

    for i in range(bins):
        mask = (preds > lower_bound[i]) * (preds <= upper_bound[i])
        if torch.any(mask):
            # print(lower_bound[i], upper_bound[i], preds[mask], torch.mean(targets[mask]))
            delta = torch.abs(torch.mean(preds[mask]) - torch.mean(targets[mask]))
            ece += delta * torch.mean(mask.float())
            mce = torch.max(mce, delta)

    return ece, mce


if __name__ == "__main__":
    p = torch.rand(1000)
    # t = torch.rand(1000)
    t = p + torch.randn(1000)
    ece_mce(p, t, bins=10)

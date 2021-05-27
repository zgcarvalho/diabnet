import torch
import numpy as np
from typing import Union, Tuple


def ece_mce(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    bins: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error
    (MCE) between predicted labels and target labels.

    Parameters
    ----------
    predictions : torch.Tensor
        A torch.ndarray or a list with predicted labels of DiabNet
    targets : torch.Tensor
        A numpy.ndarray or a list with experimental target labels
    bins : int, optional
        Number of bins, by default 10

    Returns
    -------
    ece : torch.Tensor
        Expected calibration error
    mce : torch.Tensor
        Maximum calibration error

    Reference
    ---------
    On Calibration of Modern Neural Networks (https://arxiv.org/abs/1706.04599)
    """
    # Check and convert types
    if type(predictions) not in [torch.Tensor]:
        raise TypeError("`predictions` must be a torch.Tensor.")
    if type(targets) not in [torch.Tensor]:
        raise TypeError("`targets` must be a torch.Tensor.")
    if type(bins) not in [int]:
        raise TypeError("`bins` must be a positive integer.")
    if bins <= 0:
        raise ValueError("`bins` must be a positive integer.")

    # Create lower and upper bounds
    lower_bound = torch.arange(0.0, 1.0, 1.0 / bins)
    upper_bound = lower_bound + 1.0 / bins

    # Initialize ece and mce
    ece = torch.zeros(1, device=predictions.device)
    mce = torch.zeros(1, device=predictions.device)

    # Calculate ece and mce
    for i in range(bins):
        mask = (predictions > lower_bound[i]) * (predictions <= upper_bound[i])
        if torch.any(mask):
            delta = torch.abs(torch.mean(predictions[mask]) - torch.mean(targets[mask]))
            ece += delta * torch.mean(mask.float())
            mce = torch.max(mce, delta)

    return ece, mce

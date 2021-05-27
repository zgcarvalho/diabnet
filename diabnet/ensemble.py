from typing import List
from glob import glob
import numpy as np
from diabnet.model import load, Model

__all__ = ["Ensemble"]


class Ensemble(object):
    """Class that keeps an ensemble of models loaded from "pth" files using
    their prefix.

    Ex.: prefix -> "path/model" to read all models from "path/model-000.pth" until "path/model-099.pth"
    """

    def __init__(self, prefix: str):
        """Create Ensemble object.

        Parameters
        ----------
        prefix : str
            Define prefix for files that will be loaded.
        """
        self.models = self._load(prefix)

    def _load(self, prefix: str) -> List[Model]:
        """Load parameters of trained DiabNets.

        Parameters
        ----------
        prefix : str
            Define prefix for files that will be loaded.

        Returns
        -------
        List[Model]
            A list of parameters for each model.
        """
        # Create empty list
        models = []

        # Load all model parameters based on their .pth files
        for fn in glob(f"{prefix}-???.pth"):
            models.append(load(fn))

        return models

    def apply(self, x: np.ndarray, samples_per_model: int = 1) -> np.ndarray:
        """Apply all ensemble's models to some input and return an array
        with prediction values between [0,1].

        Parameters
        ----------
        x : np.ndarray
            A numpy.ndarray with encoded features.
        samples_per_model : int, optional
            Number of samples take from each model. Only required when the
            inference has something stochastic like MC Dropout, by default 1.

        Returns
        -------
        np.ndarray
            N prediction values between [0, 1], where N is the number of models
            in the ensemble times the number of samples per model.
        """
        prob = np.array(
            [
                model.apply(x).detach().numpy()[0, 0]
                for model in self.models
                for _ in range(samples_per_model)
            ]
        )
        return prob

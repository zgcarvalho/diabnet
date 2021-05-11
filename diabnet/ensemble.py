from typing import List
from glob import glob
import numpy as np
import torch
from diabnet.model import load, Model


class Ensemble:
    """Class that keeps an ensemble of models loaded from "pth" files using their prefix.
       Ex.: prefix -> "path/model" to read all models from "path/model-000.pth" until "path/model-099.pth"
    """
    def __init__(self, prefix: str):
        self.models = self._load(prefix)

    def _load(self, prefix) -> List[Model]:
        print(prefix)
        models = []
        for fn in glob(f'{prefix}-???.pth'):
            models.append(load(fn))
        return models

    def apply(self, x: np.ndarray, samples_per_model=1) -> np.ndarray:
        """Apply all ensemble's models to some input and return an array
        with prediction values between [0,1].

        Args:
            x (np.ndarray): encoded input features
            samples_per_model (int, optional): Number of times each model is applied. Only necessary when 
            the inference has something stochastic like MC Dropout. Defaults to 1.
            age_idx (int): x column that have age information. Necessary for y correction.
            with_correction (bool): 

        Returns:
            np.ndarray: N prediction values between [0,1] where N is the number of models in the ensemble x samples_per_model.
        """
        a = np.array([m.apply(x).detach().numpy()[0,0] for m in self.models for _ in range(samples_per_model)])
        return a

if __name__ == "__main__":
    e = Ensemble('../diabnet/diabnet/models/model-sp-soft-label-positives-1000-decay')

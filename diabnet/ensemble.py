import torch
import numpy as np
from glob import glob
from diabnet.model import load



class Ensemble:
    def __init__(self, prefix, samples_per_model=10):
        self.models = self._load(prefix)
        # self.samples_per_model = samples_per_model

    def _load(self, prefix):
        print(prefix)
        models = []
        for fn in glob(f'{prefix}*.pth'):
            models.append(load(fn))
        return models

    # def forward(self, x):
    #     a = self.apply(x)
    #     print(a)
    #     return torch.mean(a)

    def apply(self, x, samples_per_model=10):
        a = np.array([torch.sigmoid(m(x)).detach().numpy()[0,0] for m in self.models for _ in range(samples_per_model)])
        # a = np.array([torch.sigmoid(m(x)/0.55).detach().numpy()[0,0] for m in self.models for _ in range(samples_per_model)])
        # print(a)
        return a

if __name__ == "__main__":
    e = Ensemble('../diabnet/diabnet/models/model-sp-soft-label-positives-1000-decay-')
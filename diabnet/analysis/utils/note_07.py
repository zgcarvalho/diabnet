import torch
import seaborn
import seaborn as sns
import numpy as np
import pandas as pd
# from typing import Dict, List, Union, Tuple, Optional
# from collections import OrderedDict
# import matplotlib.pyplot as plt
# from matplotlib.ticker import AutoMinorLocator
# from sklearn.utils.validation import indexable
from captum.attr import IntegratedGradients
from diabnet.data import encode_features


COLORS = sns.color_palette("colorblind")


def get_feature_importance(values, age, sex) -> pd.DataFrame:
    imp = values.calc_attr(age, sex)
    # testa se o SNP tem valores 1 ou 2. caso não tenha, sua importancia não pode ser calculada
    s = {k: [np.mean(imp[k]), np.median(imp[k])] for k in imp if len(imp[k]) > 0}
    df = pd.DataFrame.from_dict(s, orient="index")
    df.rename(columns={0: f"{sex}{age}_mean", 1: f"{sex}{age}_median"}, inplace=True)

    return df


class ExplainModel:
    def __init__(self, ensemble, feature_names, dataset_fn):
        self.ensemble = ensemble
        self.feat_names = feature_names
        self.dataset_fn = dataset_fn

    def encode(self, feat, age=50):
        feat[self.feat_names.index("AGE")] = age
        t = torch.Tensor(encode_features(self.feat_names, feat))
        t.requires_grad_()
        return t

    def gen_baselines(self, sex, age=50):
        baseline = np.zeros(1004, dtype=object)
        baseline[1001] = sex  # 'M'|'F'|'X'  'X' no information
        baseline = self.encode(baseline, age)
        return baseline

    def gen_inputs(self, feats, age=50):
        inputs = torch.stack([self.encode(f, age=age) for f in feats])
        return inputs

    def calc_attr(self, age, sex):
        df = pd.read_csv(self.dataset_fn)
        if sex != "X":
            df = df[df["sex"] == sex]
        features = df[self.feat_names].values
        inputs = self.gen_inputs(features, age=age)
        baseline = torch.stack(
            [self.gen_baselines(sex, age=age) for i in range(len(inputs))]
        )  # same dimension as inputs

        # attrs = []
        imp = {self.feat_names[i]: [] for i in range(1000)}
        # mask to inputs -> snps 1 and 2 == True / snps 0 == False
        mask = inputs[:, 1, :1000] > 0.5
        for m in self.ensemble.models:
            ig = IntegratedGradients(lambda x: m.apply(x))
            attr = ig.attribute(inputs, baselines=baseline)
            for i in range(1000):
                imp[self.feat_names[i]].append(attr[:, 1, i][mask[:, i]].detach().numpy())

        for i in range(1000):
            imp[self.feat_names[i]] = np.concatenate(imp[self.feat_names[i]])

        return imp

    def attr_snps_mean(self, attrs, mask):
        imp = {}
        for i in range(1000):
            imp[self.feat_names[i]] = np.concatenate([attr[:, 1, i][mask[:, i]].detach().numpy() for attr in attrs])

        return imp

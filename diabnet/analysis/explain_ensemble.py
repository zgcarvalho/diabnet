import torch
import pandas as pd
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
# from diabnet.model import load
# from diabnet.ensemble import Ensemble
from diabnet.data import get_feature_names, encode_features


class ExplainModel:
    def __init__(self, ensemble, feature_names, dataset_fn):
        self.ensemble = ensemble
        self.feat_names = feature_names
        self.dataset_fn = dataset_fn

    # def model(self, x):
    # #     # return torch.sigmoid(self.predictor(x))
    #     return self.predictor._sampler(x, samples_per_model=1)

    def encode(self, feat, age=50):
        feat[self.feat_names.index("AGE")] = age
        t = torch.Tensor(encode_features(self.feat_names, feat))
        t.requires_grad_()
        return t

    def gen_baselines(self, sex, age=50):
        baseline = np.zeros(1004, dtype=object)
        baseline[1001] = sex  # 'M'|'F'|'X'  'X' no information
        baseline = self.encode(baseline, age)
        # the lines were commented below to keep parents diagnostics as negative
        baseline.data[0, 1002] += 2.0 / 3  # remove mother diagnostic from baseline
        baseline.data[0, 1003] += 2.0 / 3  # remove mother diagnostic from baseline
        baseline.data[0, 1004] += 2.0 / 3  # remove mother diagnostic from baseline
        baseline.data[0, 1005] += 2.0 / 3  # remove father diagnostic from baseline
        baseline.data[0, 1006] += 2.0 / 3  # remove father diagnostic from baseline
        baseline.data[0, 1007] += 2.0 / 3  # remove father diagnostic from baseline
        return baseline

    def gen_inputs(self, feats, age=50):
        inputs = torch.stack([self.encode(f, age=age) for f in feats])
        return inputs

    # def calc_attr(self, age, sex):
    #     print("e ai?")
    #     df = pd.read_csv(self.dataset_fn)
    #     if sex != 'X':
    #         df = df[df['sex'] == sex]
    #     print("vamos lá")
    #     features = df[self.feat_names].values
    #     inputs = self.gen_inputs(features, age=age)
    #     print(inputs)
    #     # baseline = torch.stack([self.gen_baselines(sex, age=age) for i in range(len(inputs))])
    #     # ig = IntegratedGradients(self.model)
    #     # attr, delta = ig.attribute(inputs, baselines=baseline, return_convergence_delta=True)
    #     # # mask to inputs -> snps 1 and 2 == True / snps 0 == False
    #     # mask = inputs[:,1,:1000]>.5
    #     # return attr, mask

    def calc_attr(self, age, sex):
        # df = pd.read_csv("../datasets/visits_sp_unique_test_positivo_1000_random_0.csv")
        df = pd.read_csv(self.dataset_fn)
        if sex != "X":
            df = df[df["sex"] == sex]
        features = df[self.feat_names].values
        # labels = df["T2D"].values
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
            # print("debug")
            # print(attr)
            for i in range(1000):
                imp[self.feat_names[i]].append(attr[:, 1, i][mask[:, i]].detach().numpy())

        for i in range(1000):
            imp[self.feat_names[i]] = np.concatenate(imp[self.feat_names[i]])

        # ig = IntegratedGradients(self.model)
        # attr, delta = ig.attribute(inputs, baselines=baseline, return_convergence_delta=True)

        return imp

    def attr_snps_mean(self, attrs, mask):
        imp = {}
        for i in range(1000):
            imp[self.feat_names[i]] = np.concatenate([attr[:, 1, i][mask[:, i]].detach().numpy() for attr in attrs])

        # imp = {self.feat_names[i]:[] for i in range(1000)}
        # for attr in attrs:
        #     for i in range(1000):
        #         v = torch.mean(attr[:,1,i][mask[:,i]]).detach().numpy()
        #         if not (v != v).any():
        #             imp[self.feat_names[i]].append(v)
        # imp = {self.feat_names[k]:v for k,v in ((i, torch.mean(attr[:,1,i][mask[:,i]]).detach().numpy()) for i in range(1000)) if not (v != v).any()}

        return imp
        # return pd.DataFrame.from_dict(imp, orient='index')

    def feat_importance(self, age, sex):
        imp = self.calc_attr(age, sex)
        # testa se o SNP tem valores 1 ou 2. caso não tenha, sua importancia não pode ser calculada
        s = {k: [np.mean(imp[k]), np.median(imp[k])] for k in imp if len(imp[k]) > 0}
        df = pd.DataFrame.from_dict(s, orient="index")
        df.rename(columns={0: f"{sex}{age}_mean", 1: f"{sex}{age}_median"}, inplace=True)
        # df.sort_values(by=0, ascending=False).head(50)
        return df

    # def feat_importance(values, age, sex):
    #     imp = values.calc_attr(age, sex)
    #     # testa se o SNP tem valores 1 ou 2. caso não tenha, sua importancia não pode ser calculada
    #     s = {k: [np.mean(imp[k]), np.median(imp[k])] for k in imp if len(imp[k]) > 0}
    #     df = pd.DataFrame.from_dict(s, orient='index')
    #     df.rename(columns={0:f'{sex}{age}_mean', 1:f'{sex}{age}_median'}, inplace=True)
    #     # df.sort_values(by=0, ascending=False).head(50)
    #     return df


# PREDICTOR = load('../diabnet/models/model-sp-soft-label-positives-1000.pth')
# C = get_feature_names("../datasets/visits_sp_unique_test_positivo_1000_random_0.csv", BMI=False, sex=True, parents_diagnostics=True)

# def encode(feat, age=50):
#     feat[C.index('AGE')] = age
#     t = torch.Tensor(encode_features(C, feat))
#     t.requires_grad_()
#     return t

# def gen_baselines(sex, age=50):
#     baseline = np.zeros(1004, dtype=object)
#     baseline[1001] = sex # 'M'|'F'|'X'  'X' no information
#     baseline = encode(baseline, age)
#     baseline[0,1002] = 0.0 # remove mother diagnostic from baseline
#     baseline[0,1005] = 0.0 # remove mother diagnostic from baseline
#     return baseline

# def gen_inputs(feats, age=50, samples=1):
#     inputs = torch.stack([encode(f, age=age) for f in feats for _ in range(samples)])
#     return inputs

# def model(x):
#     return torch.sigmoid(PREDICTOR(x))

# def calc_attr(age, sex, samples):
#     df = pd.read_csv("../datasets/visits_sp_unique_test_positivo_1000_random_0.csv")
#     features = df[C].values
#     # labels = df["T2D"].values
#     inputs = gen_inputs(features, age=age, samples=samples)
#     baseline = torch.stack([gen_baselines(sex, age=age) for i in range(len(inputs))])
#     ig = IntegratedGradients(model)
#     attr, delta = ig.attribute(inputs, baselines=baseline, return_convergence_delta=True)
#     # mask to inputs -> snps 1 and 2 == True / snps 0 == False
#     mask = inputs[:,1,:1000]>.5
#     return attr, mask

# def attr_snps_mean(attr, mask):
#     imp = {C[k]:v for k,v in ((i,torch.mean(attr[:,1,i][mask[:,i]]).detach().numpy()) for i in range(1000)) if not (v != v).any()}
#     return pd.DataFrame.from_dict(imp, orient='index')


if __name__ == "__main__":
    predictor = load("../diabnet/models/model-sp-soft-label-positives-1000.pth")
    feat_names = get_feature_names(
        "../datasets/visits_sp_unique_test_positivo_1000_random_0.csv", BMI=False, sex=True, parents_diagnostics=True
    )
    e = ExplainModel(predictor, feat_names, "../datasets/visits_sp_unique_test_positivo_1000_random_0.csv")
    attr, mask = e.calc_attr(20, "X", samples=1)
    df = e.attr_snps_mean(attr, mask)
    print(df)

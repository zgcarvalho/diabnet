from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from typing import Any, List


def get_feature_names(fn: str, BMI=False, sex=True, parents_diagnostics=True) -> List[str]:
    col_names = pd.read_csv(fn).columns.values
    # always remove "id" "T2D(label)" "mo" and "fa"
    # mo_t2d and fa_t2d are removed here but can be inserted at correct position later
    col_names = np.setdiff1d(col_names, np.array(['id','T2D','mo','fa','mo_t2d', 'fa_t2d','sex','Unnamed: 0', '', ' ']), assume_unique=True)

    if not BMI:
        col_names = np.setdiff1d(col_names, np.array(['BMI']), assume_unique=True)

    features: List[str] = list(col_names)
    if sex:
        features.append("sex")

    if parents_diagnostics:
        features.append("mo_t2d")
        features.append("fa_t2d")

    return features


def encode_features(feat_names: List[str], feat_values: List[Any]) -> np.ndarray:
    _check_parents_diag(feat_names)
    m = np.zeros((2, _len_encoding(feat_names)))
    for i in range(len(feat_names)):
        if (feat_names[i] == "AGE"):
            # print("AGE == {}".format(feat_values[i]/50 ))
            m[0,i] = feat_values[i]/50 # divide to 50 to put AGE in the range ~[0,2]
            m[1,i] = 1 # works like a bias in a layer without bias
        elif (feat_names[i] == "BMI"):
            m[0,i] = feat_values[i]/10 # divide to 10 to put BMI in the range ~[0,2]
            m[1,i] = 1 # works like a bias in a layer without bias
        elif (feat_names[i] == "sex"):
            if feat_values[i] == 'F':
                m[0,i] = 2 # keep feat in the range [0,2]
            elif feat_values[i] == 'M': 
                m[1,i] = 2
        elif (feat_names[i] == "mo_t2d"):
            # ! this is HIGHLY depedent of '_fix_parents_feat'
            # if parent diagnosis are included in features both must be present simultaneously 'mo_t2d' and 'fa_t2d' 
            # AND the sequence must be 'mo_t2d' followed by 'fa_t2d' at the two last positions
            m[1,i], m[1,i+1], m[1,i+2] = 1,1,1
            if feat_values[i] == 0:
                m[0,i] = 1 # keep feat in the range [0,2]
            elif feat_values[i] == 1:
                m[0,i+1] = 1
            else:
                m[0,i+2] = 1
        elif (feat_names[i] == "fa_t2d"):
            # ! this is HIGHLY depedent of '_fix_parents_feat' and the presence of 'mo_t2d'
            # if parent diagnosis are included in features both must be present simultaneously 'mo_t2d' and 'fa_t2d' 
            # AND the sequence must be 'mo_t2d' followed by 'fa_t2d' at the two last positions
            m[1,i+2], m[1,i+3], m[1,i+4] = 1,1,1
            if feat_values[i] == 0:
                m[0,i+2] = 1
            elif feat_values[i] == 1:
                m[0,i+3] = 1
            else:
                m[0,i+4] = 1
        elif (feat_names[i][:3] == "snp"): 
            if feat_values[i] == 0:
                m[0,i] = 2
            elif feat_values[i] == 2:
                m[1,i] = 2
            else:
                m[0,i], m[1,i] = 1, 1
        else:
            # if feature name is invalid raise an exception
            raise Exception("Feature name \"{}\" is unknown".format(feat_names[i]))    
    return m


def _check_parents_diag(feat_names: List[str]):
    # check if 'mo_t2d' followed by 'fa_t2d' are at the two last positions THEY MUST BE
    if "mo_t2d" in feat_names and "fa_t2d" in feat_names:
        if feat_names[-2] != "mo_t2d" or feat_names[-1] != "fa_t2d":
            raise Exception("mo_t2d is at index {} and fa_t2d is at index {} from feat_names with {} items".format(
                        feat_names.index("mo_t2d"), feat_names.index("fa_t2d"), len(feat_names)
                    ))


def _len_encoding(feat_names: List[str]) -> int:
    if "mo_t2d" in feat_names and "fa_t2d" in feat_names:
        return len(feat_names) + 4
    else:
        return len(feat_names)


class DiabDataset(Dataset):
    def __init__(self,
                 fn_csv: str,
                 feat_names: List[str],
                 label_name="T2D",
                 soft_label=True,
                 soft_label_alpha=0.15,
                 soft_label_baseline=0.0,
                 device='cuda'):
        dt = pd.read_csv(fn_csv)
        _check_parents_diag(feat_names)
        self.feat_names = feat_names
        self.n_feat = _len_encoding(feat_names)

        # soft label gives values greater than -1 to younger negatives (uncertainty)
        if soft_label:
            self.labels = self._soft_negative_label_adjusted_by_age(
                dt["AGE"].values,
                dt[label_name].values,
                alpha=soft_label_alpha,
                baseline=soft_label_baseline)
            # self.labels = self._soft_label(dt[label_name].values, alpha=soft_label_alpha)
        else:
            self.labels = dt[label_name].values
        self.labels = torch.unsqueeze(torch.tensor(self.labels, dtype=torch.float), 1).to(device)
        self.raw_values = dt[feat_names].values
        self.features = torch.tensor([encode_features(self.feat_names, raw_value) for raw_value in self.raw_values], dtype=torch.float).to(device)

    @staticmethod
    def _soft_negative_label_adjusted_by_age(ages: np.ndarray, labels: np.ndarray, alpha: float, baseline: float) -> np.ndarray:
        soft_labels = np.zeros(len(labels))
        for (i,age) in enumerate(ages):
            if labels[i] == 1:
                soft_labels[i] = 1
            else:
                if age < 20:
                    soft_labels[i] = alpha # penalty to uncertanty
                elif age > 80:
                    soft_labels[i] = 0 
                else:
                    soft_labels[i] = alpha * (1 - (age-20)/(80-20))
                soft_labels[i] += baseline
        return soft_labels

    @staticmethod
    def _soft_label(labels: np.ndarray, alpha: float) -> np.ndarray:
        soft_labels = np.abs(labels-alpha)
        # soft_labels = np.maximum(labels, alpha)
        return soft_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


    # def randomize_age(self, idx):
    #     true_age = self.features[idx][0,-1]
    #     if self.labels[idx] == 0:
    #         if true_age > 70:
    #             age = true_age
    #         else:
    #             age = np.random.randint(true_age, 70+1)
    #         # age = true_age + np.abs(np.random.normal(0, 3))
    #     else:
    #         if true_age < 30:
    #             age = true_age
    #         else: 
    #             age = np.random.randint(30, true_age+1)
    #         # age = true_age - np.abs(np.random.normal(0, 3))
        
    #     for i in range(self.n_feat):
    #         if self.feat_names[i] == "AGE":
    #             self.features[idx][0,i] = age


# class CalibrationDataset(Dataset):
#     def __init__(self,
#                  fn_csv: str,
#                  feat_names: List[str],
#                  balance=True,
#                  label_name="T2D",
#                  age_cutoff=60,
#                  device='cuda'):

#         dt = pd.read_csv(fn_csv)
#         dt = dt[dt['AGE'] >= age_cutoff]
#         if balance:
#             dt = self.balance_df(dt)
#         _check_parents_diag(feat_names)
#         self.feat_names = feat_names
#         self.n_feat = _len_encoding(feat_names)
         

#         self.labels = dt[label_name].values
#         self.labels = torch.unsqueeze(torch.tensor(self.labels, dtype=torch.float), 1).to(device)
            
#         self.raw_values = dt[feat_names].values
#         self.features = torch.tensor([encode_features(self.feat_names, raw_value) for raw_value in self.raw_values], dtype=torch.float).to(device)

#     @staticmethod
#     def balance_df(df: pd.DataFrame) -> pd.DataFrame:
#         # print(df)
#         g = df.groupby(['T2D'])
#         g = g.apply(lambda x: x.sample(g.size().min()))
#         # print(g)
#         return g
        
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]


if __name__ == "__main__":
    DATASET = '../datasets/visits_sp_unique_train_positivo_1000_negative_0.csv'
    # def get_feature_names(fn, BMI=False):
    #     import pandas as pd 
    #     import numpy as np
    #     col_names = pd.read_csv(fn).columns.values
    #     # features = np.delete(col_names, ['id','BMI','T2D'])
    #     if BMI:
    #         features = list(np.setdiff1d(col_names, np.array(['id','T2D','mo','fa']), assume_unique=True))
    #     else:
    #         features = list(np.setdiff1d(col_names, np.array(['id','BMI','T2D','mo','fa']), assume_unique=True))
        
    #     if "mo_t2d" in features and "fa_t2d" in features:
    #         #put mo_t2d at the end and add 2 positions to n_feat (mo_t2d needs 3 positions in input vector)
    #         features.remove("mo_t2d")
    #         features.append("mo_t2d")
    #         features.remove("fa_t2d")
    #         features.append("fa_t2d")
            
    #     return features
    
    features = get_feature_names(DATASET, BMI=False)[991:]
    # print(features)
    dataset = DiabDataset(DATASET, features, soft_label=True)
    print(dataset.features.shape)
    print(dataset.labels.shape)
    # print(dataset.raw_values.shape)
    # print(dataset.raw_values[0])
    print(dataset[0])
    # print(dataset.features[-1])
    # print(dataset.raw[-1])

    # c = ['snp_{}'.format(i) for i in range(1,101)]
    # c.append('AGE')
    # d = DiabDataset('../datasets/train.csv', c, label='T2D', random_age=True)
    # print(d.features)
    # print(d.n_feat)
    # print(d[1])
    # print(len(d))
    # print(d[0][0].shape)

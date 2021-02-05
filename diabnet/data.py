from typing import Any, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def get_feature_names(fn_csv: str, use_bmi=False, use_sex=True, use_parents_diagnosis=True) -> List[str]:
    '''
    Get feature names from a CSV data file (header). IDs, label and unnamed columns are deleted.

    Args:
        fn_csv (str): CSV filename with data. First line must have feature names.
        use_bmi (bool): use feature named "BMI". Default is False.
        use_sex (bool): use sex as a feature. Default is True.
        use_parents_diagnosis: use parents' diagnosis as feature. Default is True.

    Returns:
        List with all feature names that will be used by the model. (List[str])

    '''

    col_names = pd.read_csv(fn_csv).columns.values
    # always remove "id" "famid" "T2D(label)" "mo" and "fa"
    # mo_t2d and fa_t2d are removed here but can be inserted into the correct position later
    col_names = np.setdiff1d(
        col_names,
        np.array(['id','famid','T2D','mo','fa','mo_t2d','fa_t2d','BMI','sex','Unnamed: 0','',' ']),
        assume_unique=True)

    features: List[str] = list(col_names)
    if use_bmi:
        features.append('BMI')

    if use_sex:
        features.append('sex')

    if use_parents_diagnosis:
        features.append('mo_t2d')
        features.append('fa_t2d')

    return features


def encode_features(feat_names: List[str], feat_values: List[Any]) -> np.ndarray:
    '''
    Encode features into a matrix to be used as input.
    Args:
        feat_names (List[str]):
        feat_values (List[Any]):

    Raises:
        Unknown feature name.

    Returns:
        Matrix (np.ndarray) with shape (2, len_encoded_features) with feature values
        to be used as input.

    '''
    _check_parents_diag(feat_names)
    m = np.zeros((2, _len_encoding(feat_names)))
    for i, feat_name in enumerate(feat_names):
        if feat_name == 'AGE':
            # print("AGE == {}".format(feat_values[i]/50 ))
            m[0,i] = feat_values[i]/50 # divide to 50 to put AGE in the range ~[0,2]
            m[1,i] = 1 # works like a bias in a layer without bias
        elif feat_name == 'BMI':
            m[0,i] = feat_values[i]/10 # divide to 10 to put BMI in the range ~[0,2]
            m[1,i] = 1 # works like a bias in a layer without bias
        elif feat_name == 'sex':
            if feat_values[i] == 'F':
                m[0,i] = 2 # keep feat in the range [0,2]
            elif feat_values[i] == 'M':
                m[1,i] = 2
        elif feat_name == 'mo_t2d':
            # ! this is HIGHLY depedent of '_fix_parents_feat'
            # if parent diagnosis are included in features both 'mo_t2d' and 'fa_t2d' must be present
            # AND the sequence must be 'mo_t2d' followed by 'fa_t2d' at the two last positions
            m[1,i], m[1,i+1], m[1,i+2] = 1,1,1 # works like a bias
            if feat_values[i] == 0:
                m[0,i] = 1 # keep feat in the range [0,2]
            elif feat_values[i] == 1:
                m[0,i+1] = 1
            else:
                m[0,i+2] = 1
        elif feat_name == 'fa_t2d':
            # ! this is HIGHLY depedent of '_fix_parents_feat' and the presence of 'mo_t2d'
            # if parent diagnosis are included in features both 'mo_t2d' and 'fa_t2d' must be present
            # AND the sequence must be 'mo_t2d' followed by 'fa_t2d' at the two last positions
            m[1,i+2], m[1,i+3], m[1,i+4] = 1,1,1 # works like a bias
            if feat_values[i] == 0:
                m[0,i+2] = 1
            elif feat_values[i] == 1:
                m[0,i+3] = 1
            else:
                m[0,i+4] = 1
        elif feat_name[:3] == 'snp':
            if feat_values[i] == 0:
                m[0,i] = 2
            elif feat_values[i] == 2:
                m[1,i] = 2
            else:
                m[0,i], m[1,i] = 1, 1
        else:
            # if feature name is invalid raise an exception
            raise Exception(f'Feature name "{feat_name}" is unknown')
    return m


def _check_parents_diag(feat_names: List[str]):
    # check if 'mo_t2d' followed by 'fa_t2d' are at the two last positions THEY MUST BE
    if "mo_t2d" in feat_names and "fa_t2d" in feat_names:
        if feat_names[-2] != "mo_t2d" or feat_names[-1] != "fa_t2d":
            raise Exception(
                "mo_t2d is at index {} and fa_t2d is at index {} from \
                feat_names with {} items".format(
                    feat_names.index("mo_t2d"),
                    feat_names.index("fa_t2d"),
                    len(feat_names)))


def _len_encoding(feat_names: List[str]) -> int:
    # mo_t2d and fa_t2d use 3 columns (+2 each one = 4)
    if "mo_t2d" in feat_names and "fa_t2d" in feat_names:
        return len(feat_names) + 4

    return len(feat_names)


class DiabDataset(Dataset):
    '''

    Args:
        fn_csv (str): CSV filename with data
        feat_names (List[str]): feature names that will be used
        label_name (str): feature name (csv column name) that identify the label
        soft_label (bool): use soft label. Prob_negatives > 0 and Prob_positives < 1. Default: True
        soft_label_baseline (float): value to Prob_negatives. Default: 0.0
        soft_label_topline (float): value to Prob_positives. Default: 1.0
        soft_label_baseline_slope (float): decrease in Prob_negatives uncertainty. Unevenness between 20yo and 80 yo. Default: 0.0
        device (str): tensor device. Default is 'cuda'.

    Attributes:
        feat_names (List[str]):
        n_feat (int):
        raw_values (np.ndarray):
        features (tensor):
        labels (tensor):

    '''
    def __init__(self,
                 fn_csv: str,
                 feat_names: List[str],
                 label_name='T2D',
                 soft_label=True,
                 soft_label_baseline=0.0,
                 soft_label_topline=1.0,
                 soft_label_baseline_slope=0.0,
                 device='cuda'):

        df = pd.read_csv(fn_csv)
        _check_parents_diag(feat_names)

        self.feat_names = feat_names
        self.n_feat = _len_encoding(feat_names)
        self._raw_values = df[feat_names].values
        self._raw_labels = df[label_name].values
        self._ages = df['AGE'].values
        self.features = torch.tensor(
            [encode_features(self.feat_names, raw_value) for raw_value in self._raw_values],
            dtype=torch.float).to(device)
        
        if soft_label:
            labels = self._soft_label(
                baseline=soft_label_baseline,
                topline=soft_label_topline,
                baseline_slope=soft_label_baseline_slope)
        else:
            labels = self._raw_labels
        self.labels = torch.unsqueeze(torch.tensor(labels, dtype=torch.float), 1).to(device)

    def _soft_label(self, baseline: float, topline: float, baseline_slope: float) -> np.ndarray:
        assert baseline < topline and baseline_slope <= 0.0
        if baseline_slope == 0.0:
            soft_labels = np.maximum(self._raw_labels * topline, baseline)
        else:
            ages = np.minimum(np.maximum(self._ages, 20.0), 80.0)
            base_adjusted_by_age = np.maximum(baseline + baseline_slope * (ages - 20.0)/(80.0 - 20.0), 0)
            soft_labels = np.maximum(self._raw_labels * topline, base_adjusted_by_age)
        return soft_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def ajust_soft_label(self, baseline, topline, baseline_slope):
        labels = self._soft_label(baseline, topline, baseline_slope)
        self.labels = torch.unsqueeze(torch.tensor(labels, dtype=torch.float), 1).to('cuda')

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

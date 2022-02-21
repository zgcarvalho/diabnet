from typing import List
import pandas as pd
import numpy as np
# from typing import List
import calibration

# import torch
# import matplotlib.pyplot as plt
import seaborn as sns
# from collections import OrderedDict
# from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import (
    roc_auc_score,
    # roc_curve,
    # precision_recall_curve,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    average_precision_score,
    precision_score,
    recall_score,  # if pos_label = (sensitivity or TPR) if neg_label = (specificity or TNR)
    confusion_matrix,
)
# import sys
from diabnet.apply_ensemble import Predictor
from diabnet.data import get_feature_names

sns.set()

# sys.path.append("../")
# from diabnet.model import load
# from diabnet.apply_ensemble import Predictor
# from diabnet.data import get_feature_names

# from diabnet.calibration import ece_mce
# from astropy.stats import bootstrap
# from scipy.stats import rankdata

# from scipy.signal import wiener
# from scipy.interpolate import interp1d


COLORS = sns.color_palette("colorblind")
sns.set_style("whitegrid")
sns.set_style("ticks", {"axes.grid": True, "grid.color": ".95", "grid.linestyle": "-"})


def ece(targets, preds, bins=5):
    """ECE - Expected Calibration Error
    Ref: On Calibration of Modern Neural Networks (https://arxiv.org/abs/1706.04599)
    """
    lower_bounds = np.arange(0.0, 1.0, 1.0 / bins)
    upper_bounds = lower_bounds + 1.0 / bins

    ece = 0.0

    for i in range(bins):
        mask = (preds > lower_bounds[i]) * (preds <= upper_bounds[i])
        if np.any(mask):
            delta = np.abs(np.mean(preds[mask]) - np.mean(targets[mask]))
            ece += delta * np.mean(mask)

    return ece


def ece_(targets, preds, bins=10):
    """ECE - Expected Calibration Error
    Ref: Verified Uncertainty Calibration https://arxiv.org/abs/1909.10155
    """
    return calibration.get_ece(preds, targets, num_bins=bins)


def mce(targets, preds, bins=5):
    """MCE - Maximum Calibration Error
    Ref: On Calibration of Modern Neural Networks (https://arxiv.org/abs/1706.04599)
    """
    lower_bounds = np.arange(0.0, 1.0, 1.0 / bins)
    upper_bounds = lower_bounds + 1.0 / bins

    mce = 0.0

    for i in range(bins):
        mask = (preds > lower_bounds[i]) * (preds <= upper_bounds[i])
        if np.any(mask):
            delta = np.abs(np.mean(preds[mask]) - np.mean(targets[mask]))
            mce = np.maximum(mce, delta)

    return mce


class Dataset:
    def __init__(self, fn: str, feat_names: List[str], predictor: Predictor):
        self.df = pd.read_csv(fn)
        self._feat_names = feat_names
        self._predictor = predictor
        # properties below are computed when needed (lazy)
        self._features = None
        self._labels = None
        self._predictions = None
        self._predictions_ensemble = None
        self._predictions_per_age = None
        self._predictions_per_age_ensemble = None
        self._predictions_years_before = None
        self._age_range = range(20, 80, 5)

    @property
    def features(self):
        if self._features is None:
            self._features = self.df[self._feat_names].values
        return self._features

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self.df["T2D"].values
        return self._labels

    @property
    def predictions(self):
        # returns the mean value from ensemble predictions
        if self._predictions is None:
            self._predictions = np.array([np.mean(preds) for preds in self.predictions_ensemble])
        return self._predictions

    @property
    def predictions_ensemble(self):
        # Returns all output values from ensemble models. One for each model.
        # Output shape is NxM where N is number of patients and M is number of models into the ensemble
        if self._predictions_ensemble is None:
            self._predictions_ensemble = np.array(
                [self._predictor.patient(f, age=-1, samples_per_model=1) for f in self.features]
            )
        return self._predictions_ensemble

    @property
    def predictions_per_age(self):
        if self._predictions_per_age is None:
            self._predictions_per_age = {
                age: np.array([np.mean(preds) for preds in self.predictions_per_age_ensemble[age]])
                for age in self.predictions_per_age_ensemble.keys()
            }
        return self._predictions_per_age

    @property
    def predictions_per_age_ensemble(self):
        if self._predictions_per_age_ensemble is None:
            self._predictions_per_age_ensemble = {
                age: np.array([self._predictor.patient(f, age=age, samples_per_model=1) for f in self.features])
                for age in self._age_range
                # for age in range(20, 80, 5)
            }
        return self._predictions_per_age_ensemble

    @property
    def predictions_years_before(self):
        # Probably this analysis make sense only in patients with known first diagnosis
        i = self._feat_names.index("AGE")
        if self._predictions is None:
            self._predictions_years_before = {
                yb: np.array(
                    [np.mean(self._predictor.patient(f, age=f[i] - yb, samples_per_model=1)) for f in self.features]
                )
                for yb in range(0, 10, 1)
            }
        return self._predictions_years_before

    def bootstrap(self, func, num=2000, alpha=0.05, interval="CI"):
        value = func(self.labels, self.predictions)
        n_patients = len(self.labels)
        n_models = len(self.predictions_ensemble[0])
        data = np.empty(num)
        for i in range(num):
            mask_preds = np.random.choice(n_patients, n_patients)
            mask_ensemble = np.random.choice(n_models, n_models)
            data[i] = func(
                self.labels[mask_preds],
                np.mean(self.predictions_ensemble[mask_preds][:, mask_ensemble], 1),
            )
        data = np.sort(data)
        if interval == "CI":
            return {
                "value": value,
                "mean": np.mean(data),
                "lower": np.percentile(data, alpha * 100.0 / 2.0),
                "upper": np.percentile(data, 100.0 - alpha * 100.0 / 2.0),
                "interval": interval,
            }
        elif interval == "HDI":
            lower, upper = self._hdi(data, 1.0 - alpha)
            return {
                "value": value,
                "mean": np.mean(data),
                "lower": lower,
                "upper": upper,
                "interval": interval,
            }

    @staticmethod
    def _hdi(data, p):
        """Highest Density Interval"""
        j = int(len(data) * p)
        diff, lower, upper = data[-1] - data[0], data[0], data[-1]
        for i in range(len(data) - j):
            new_diff = data[i + j - 1] - data[i]
            if diff > new_diff:
                diff = new_diff
                lower = data[i]
                upper = data[i + j - 1]
        return (lower, upper)

    def _hdi_idx(data, p):
        """Highest Density Interval"""
        j = int(len(data) * p)
        diff = data[-1] - data[0]
        for i in range(len(data) - j):
            new_diff = data[i + j - 1] - data[i]
            if diff > new_diff:
                diff = new_diff
                lower_idx = i
                upper_idx = i + j - 1
        return (lower_idx, upper_idx)


class DiabNetReport:
    def __init__(
        self,
        ensemble,
        data_suffix,
        use_negatives=True,
        use_sex=True,
        use_parents=True,
    ):
        # data_suffix -> e.g. "positivo_1000_random_0.csv"
        self.data_suffix = data_suffix

        self.feat_names = get_feature_names(
            "../data/datasets/visits_sp_unique_test_" + data_suffix,
            # use_bmi=use_bmi,
            use_sex=use_sex,
            use_parents_diagnosis=use_parents,
        )
        if use_negatives:
            negatives_csv = "../data/datasets/visits_sp_unique_test_positivo_1000_random_0_negatives_older60.csv"
        else:
            negatives_csv = None

        self.predictor = Predictor(ensemble, self.feat_names, negatives_csv=negatives_csv)
        self._negatives_older60 = None
        # all dateset below are prepared only when needed (lazy)
        self._dataset_test_unique = None
        self._dataset_test_unique_subset_neg_older50 = None
        self._dataset_test_changed = None
        self._dataset_test_first_diag = None
        # self._dataset_test_all = None

    @property
    def negatives_older60(self):
        if self._negatives_older60 is None:
            self._negatives_older60 = self.predictor.negatives_life(
                age_range=range(20, 80, 5), bmi=-1, samples_per_model=10
            )
        return self._negatives_older60

    @property
    def dataset_test_unique(self):
        if self._dataset_test_unique is None:
            self._dataset_test_unique = Dataset(
                "../data/datasets/visits_sp_unique_test_" + self.data_suffix,
                self.feat_names,
                self.predictor,
            )
        return self._dataset_test_unique

    @property
    def dataset_test_unique_subset_neg_older50(self):
        # subset positives or negatives older than 50
        if self._dataset_test_unique_subset_neg_older50 is None:
            db = Dataset(
                "../data/datasets/visits_sp_unique_test_" + self.data_suffix,
                self.feat_names,
                self.predictor,
            )
            db.df = db.df[(db.df.T2D > 0.5) | (db.df.AGE > 50)]
            self._dataset_test_unique_subset_neg_older50 = db
        return self._dataset_test_unique_subset_neg_older50

    @property
    def dataset_test_changed(self):
        if self._dataset_test_changed is None:
            self._dataset_test_changed = Dataset(
                "../data/datasets/visits_sp_changed_test_" + self.data_suffix,
                self.feat_names,
                self.predictor,
            )
        return self._dataset_test_changed

    @property
    def dataset_test_first_diag(self):
        # selects a subset from "changed" where is possible to infer the age that a pacient
        # received the first positive diagnosis
        if self._dataset_test_first_diag is None:
            db = Dataset(
                "../data/datasets/visits_sp_changed_test_" + self.data_suffix,
                self.feat_names,
                self.predictor,
            )
            # take the negative rows, group them by patient id and take the greatest age
            neg = db.df[db.df.T2D == 0][["id", "AGE"]].groupby("id").max()
            # take the postive rows, group them by patient id and take the lowest age
            pos = db.df[db.df.T2D == 1][["id", "AGE"]].groupby("id").min()
            # join both selections by id
            dtmp = neg.join(pos, lsuffix="_neg", rsuffix="_pos")
            # into the "changed" dataset there are patients with negative diagnosis after a positive.
            # As I don't know how to interpret this correctly I think better to keep this patients out of this subset
            dtmp = dtmp[dtmp.AGE_neg < dtmp.AGE_pos]
            tmp = pd.concat(
                [
                    pd.DataFrame({"AGE": dtmp.AGE_neg}),
                    pd.DataFrame({"AGE": dtmp.AGE_pos}),
                ]
            )
            tmp = tmp.reset_index()
            db.df = db.df.merge(tmp, on=["id", "AGE"], how="inner")
            self._dataset_test_first_diag = db
        return self._dataset_test_first_diag

    # this subset has more than one entry for some patients. this entries are at different ages
    # and sometimes has different diagnosis. it is not being used now.
    # @property
    # def dataset_test_all(self):
    #     if self._dataset_test_all is None:
    #         self._dataset_test_all = Dataset(
    #             "../data/datasets/visits_sp_all_test_" + self.data_suffix,
    #             self.feat_names,
    #             self.predictor,
    #         )
    #     return self._dataset_test_all

    ######################################################
    # Report methods
    @staticmethod
    def _auc(db, interval, bootnum, alpha):
        return db.bootstrap(roc_auc_score, num=bootnum, alpha=alpha, interval=interval)

    def auc(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._auc(self.dataset_test_unique, interval, bootnum, alpha)

    def auc_neg_older50(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._auc(self.dataset_test_unique_subset_neg_older50, interval, bootnum, alpha)

    @staticmethod
    def _average_precision(db, interval, bootnum, alpha):
        return db.bootstrap(average_precision_score, num=bootnum, alpha=alpha, interval=interval)

    def average_precision(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._average_precision(self.dataset_test_unique, interval, bootnum, alpha)

    def average_precision_neg_older50(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._average_precision(self.dataset_test_unique_subset_neg_older50, interval, bootnum, alpha)

    @staticmethod
    def _brier(db, interval, bootnum, alpha):
        return db.bootstrap(brier_score_loss, num=bootnum, alpha=alpha, interval=interval)

    def brier(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._brier(self.dataset_test_unique, interval, bootnum, alpha)

    def brier_neg_older50(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._brier(self.dataset_test_unique_subset_neg_older50, interval, bootnum, alpha)

    def brier_ensemble(self):
        return brier_score_loss(
            self.dataset_test_unique.labels.repeat(100), self.dataset_test_unique.predictions_ensemble.flatten()
        )

    @staticmethod
    def _f1(db, interval, bootnum, alpha):
        return db.bootstrap(
            lambda labels, preds: f1_score(labels, np.round(preds)),
            num=bootnum,
            alpha=alpha,
            interval=interval,
        )

    def f1(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._f1(self.dataset_test_unique, interval, bootnum, alpha)

    def f1_neg_older50(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._f1(self.dataset_test_unique_subset_neg_older50, interval, bootnum, alpha)

    @staticmethod
    def _acc(db, interval, bootnum, alpha):
        return db.bootstrap(
            lambda labels, preds: accuracy_score(labels, np.round(preds)),
            num=bootnum,
            alpha=alpha,
            interval=interval,
        )

    def acc(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._acc(self.dataset_test_unique, interval, bootnum, alpha)

    def acc_neg_older50(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._acc(self.dataset_test_unique_subset_neg_older50, interval, bootnum, alpha)

    @staticmethod
    def _bacc(db, interval, bootnum, alpha):
        return db.bootstrap(
            lambda labels, preds: balanced_accuracy_score(labels, np.round(preds)),
            num=bootnum,
            alpha=alpha,
            interval=interval,
        )

    def bacc(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._bacc(self.dataset_test_unique, interval, bootnum, alpha)

    def bacc_neg_older50(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._bacc(self.dataset_test_unique_subset_neg_older50, interval, bootnum, alpha)

    @staticmethod
    def _precision(db, interval, bootnum, alpha):
        return db.bootstrap(
            lambda labels, preds: precision_score(labels, np.round(preds)),
            num=bootnum,
            alpha=alpha,
            interval=interval,
        )

    def precision(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._precision(self.dataset_test_unique, interval, bootnum, alpha)

    def precision_neg_older50(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._precision(self.dataset_test_unique_subset_neg_older50, interval, bootnum, alpha)

    @staticmethod
    def _recall(db, interval, bootnum, alpha, pos_label):
        return db.bootstrap(
            lambda labels, preds: recall_score(labels, np.round(preds), pos_label=pos_label),
            num=bootnum,
            alpha=alpha,
            interval=interval,
        )

    def sensitivity(self, interval="CI", bootnum=5000, alpha=0.05):
        # sensitivity is the same as positive class recall
        return self._recall(self.dataset_test_unique, interval, bootnum, alpha, pos_label=1)

    def sensitivity_neg_older50(self, interval="CI", bootnum=5000, alpha=0.05):
        # sensitivity is the same as positive class recall
        return self._recall(self.dataset_test_unique_subset_neg_older50, interval, bootnum, alpha, pos_label=1)

    def specificity(self, interval="CI", bootnum=5000, alpha=0.05):
        # specificity is the same as negative class recall
        return self._recall(self.dataset_test_unique, interval, bootnum, alpha, pos_label=0)

    def specificity_neg_older50(self, interval="CI", bootnum=5000, alpha=0.05):
        # specificity is the same as negative class recall
        return self._recall(self.dataset_test_unique_subset_neg_older50, interval, bootnum, alpha, pos_label=0)

    @staticmethod
    def _ece(db, interval, bootnum, alpha):
        return db.bootstrap(ece, num=bootnum, alpha=alpha, interval=interval)

    def ece(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._ece(self.dataset_test_unique, interval, bootnum, alpha)

    def ece_neg_older50(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._ece(self.dataset_test_unique_subset_neg_older50, interval, bootnum, alpha)

    @staticmethod
    def _mce(db, interval, bootnum, alpha):
        return db.bootstrap(mce, num=bootnum, alpha=alpha, interval=interval)

    def mce(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._mce(self.dataset_test_unique, interval, bootnum, alpha)

    def mce_neg_older50(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._mce(self.dataset_test_unique_subset_neg_older50, interval, bootnum, alpha)

    @staticmethod
    def _confusion(db, interval, bootnum, alpha):
        # return db.bootstrap(
        #     lambda labels, preds: confusion_matrix(labels, np.round(preds), normalize='true'),
        #     num=bootnum,
        #     alpha=alpha,
        #     interval=interval
        # )
        return confusion_matrix(db.labels, np.round(db.predictions))

    def confusion(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._confusion(self.dataset_test_unique, interval, bootnum, alpha)

    def confusion_neg_older50(self, interval="CI", bootnum=5000, alpha=0.05):
        return self._confusion(self.dataset_test_unique_subset_neg_older50, interval, bootnum, alpha)

    ##########

    def auc_per_age(self):
        db = self.dataset_test_unique
        aucs = {age: roc_auc_score(db.labels, preds) for age, preds in db.predictions_per_age.items()}
        for k, v in aucs.items():
            print("Predictions using patient age = {} have AUC = {:.3f}".format(k, v))

    def avgprec_per_age(self):
        db = self.dataset_test_unique
        aucs = {age: roc_auc_score(db.labels, preds) for age, preds in db.predictions_per_age.items()}
        for k, v in aucs.items():
            print("Predictions using patient age = {} have AUC = {:.3f}".format(k, v))

    def f1_per_age(self):
        db = self.dataset_test_unique
        f1s = {age: f1_score(db.labels, np.round(preds)) for age, preds in db.predictions_per_age.items()}
        for k, v in f1s.items():
            print("Predictions using patient age = {} have F1-Score = {:.3f}".format(k, v))

    def acc_per_age(self):
        db = self.dataset_test_unique
        accs = {age: accuracy_score(db.labels, np.round(preds)) for age, preds in db.predictions_per_age.items()}
        for k, v in accs.items():
            print("Predictions using patient age = {} have Accuracy = {:.3f}".format(k, v))

    def bacc_per_age(self):
        db = self.dataset_test_unique
        baccs = {
            age: balanced_accuracy_score(db.labels, np.round(preds)) for age, preds in db.predictions_per_age.items()
        }
        for k, v in baccs.items():
            print("Predictions using patient age = {} have Balanced Accuracy = {:.3f}".format(k, v))

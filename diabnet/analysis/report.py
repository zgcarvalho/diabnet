from typing import List
import pandas as pd
import numpy as np
# import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from matplotlib.ticker import AutoMinorLocator

sns.set()
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    average_precision_score,
    precision_score,
    recall_score,  # if pos_label = (sensitivity or TPR) if neg_label = (specificity or TNR)
)
import sys

sys.path.append("../")
# from diabnet.model import load
from diabnet.apply_ensemble import Predictor
from diabnet.data import get_feature_names

# from diabnet.calibration import ece_mce
from astropy.stats import bootstrap
from scipy.stats import rankdata
# from scipy.signal import wiener
# from scipy.interpolate import interp1d
from typing import List
import calibration

COLORS = sns.color_palette("colorblind")
sns.set_style("whitegrid")
sns.set_style("ticks", {"axes.grid": True, "grid.color": ".95", "grid.linestyle": "-"})


def ece(targets, preds, bins=10):
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


def mce(targets, preds, bins=10):
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
            self._predictions = np.array(
                [np.mean(preds) for preds in self.predictions_ensemble]
            )
        return self._predictions

    @property
    def predictions_ensemble(self):
        # Returns all output values from ensemble models. One for each model. 
        # Output shape is NxM where N is number of patients and M is number of models into the ensemble
        if self._predictions_ensemble is None:
            self._predictions_ensemble = np.array(
                [
                    self._predictor.patient(f, age=-1, samples_per_model=1)
                    for f in self.features
                ]
            )
        return self._predictions_ensemble

    @property
    def predictions_per_age(self):
        if self._predictions_per_age is None:
            self._predictions_per_age = {
                age: np.array(
                    [np.mean(preds) for preds in self.predictions_per_age_ensemble[age]]
                )
                for age in self.predictions_per_age_ensemble.keys()
            }
        return self._predictions_per_age

    @property
    def predictions_per_age_ensemble(self):
        if self._predictions_per_age_ensemble is None:
            self._predictions_per_age_ensemble = {
                age: np.array(
                    [
                        self._predictor.patient(f, age=age, samples_per_model=1)
                        for f in self.features
                    ]
                )
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
                    [
                        np.mean(
                            self._predictor.patient(
                                f, age=f[i] - yb, samples_per_model=1
                            )
                        )
                        for f in self.features
                    ]
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
        l = int(len(data) * p)
        diff, lower, upper = data[-1] - data[0], data[0], data[-1]
        for i in range(len(data) - l):
            new_diff = data[i + l - 1] - data[i]
            if diff > new_diff:
                diff = new_diff
                lower = data[i]
                upper = data[i + l - 1]
        return (lower, upper)

    def _hdi_idx(data, p):
        """Highest Density Interval"""
        l = int(len(data) * p)
        diff = data[-1] - data[0]
        for i in range(len(data) - l):
            new_diff = data[i + l - 1] - data[i]
            if diff > new_diff:
                diff = new_diff
                lower_idx = i
                upper_idx = i + l -1
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
            negatives_csv = (
                "../data/datasets/visits_sp_unique_test_positivo_1000_random_0_negatives_older60.csv"
            )
        else:
            negatives_csv = None

        self.predictor = Predictor(
            ensemble, self.feat_names, negatives_csv=negatives_csv
        )
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
            db.df = db.df[(db.df.T2D > 0.5) | (db.df.AGE >= 50)]
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
        if self._dataset_test_first_diag == None:
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
            # into the "changed" dataset there are patients with negative diagnosis after a positive. As I don't know how to interpret this
            # correctly I think better to keep this patients out of this subset
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
        return brier_score_loss(self.dataset_test_unique.labels.repeat(100), self.dataset_test_unique.predictions_ensemble.flatten())


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

    ##########
    def auc_per_age(self):
        db = self.dataset_test_unique
        aucs = {
            age: roc_auc_score(db.labels, preds)
            for age, preds in db.predictions_per_age.items()
        }
        for k, v in aucs.items():
            print("Predictions using patient age = {} have AUC = {:.3f}".format(k, v))

    def f1_per_age(self):
        db = self.dataset_test_unique
        f1s = {
            age: f1_score(db.labels, np.round(preds))
            for age, preds in db.predictions_per_age.items()
        }
        for k, v in f1s.items():
            print(
                "Predictions using patient age = {} have F1-Score = {:.3f}".format(k, v)
            )

    def acc_per_age(self):
        db = self.dataset_test_unique
        accs = {
            age: accuracy_score(db.labels, np.round(preds))
            for age, preds in db.predictions_per_age.items()
        }
        for k, v in accs.items():
            print(
                "Predictions using patient age = {} have Accuracy = {:.3f}".format(k, v)
            )

    def bacc_per_age(self):
        db = self.dataset_test_unique
        baccs = {
            age: balanced_accuracy_score(db.labels, np.round(preds))
            for age, preds in db.predictions_per_age.items()
        }
        for k, v in baccs.items():
            print(
                "Predictions using patient age = {} have Balanced Accuracy = {:.3f}".format(
                    k, v
                )
            )

    # def roc_curves(self):
    #     db = self.dataset_test_unique
    #     for _, pr in db.predictions_per_age.items():
    #         fpr, tpr, threshold = roc_curve(db.labels, pr)
    #         plt.plot(fpr, tpr)
    #     plt.ylim(0, 1)
    #     plt.xlim(0, 1)
    #     # axis labels
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.show()

    # def roc_ci(self, ci=95, bootstrap_samples=1000):
    #     self.roc_CI(self.dataset_test_unique, ci, bootstrap_samples)

    # def roc_ci_older50(self, ci=95, bootstrap_samples=1000):
    #     self.roc_CI(self.dataset_test_unique_subset_older50, ci, bootstrap_samples)



    # @staticmethod
    # def roc_CI(db, ci, bootstrap_samples):
    #     plt.figure(figsize=(6, 6), dpi=300)
    #     # db = self.dataset_test_unique

    #     label_pred = np.stack((db.labels, db.predictions), axis=1)
    #     b = bootstrap(label_pred, bootstrap_samples)
    #     rcs = [roc_auc_score(x[:, 0], x[:, 1]) for x in b]
    #     idx_sorted = rankdata(rcs, method="ordinal")

    #     idx_min = int(0.05 * bootstrap_samples)
    #     idx_max = int(0.95 * bootstrap_samples)

    #     for i in range(len(idx_sorted)):
    #         if idx_sorted[i] >= idx_min and idx_sorted[i] <= idx_max:
    #             fpr, tpr, _ = roc_curve(b[i][:, 0], b[i][:, 1])

    #             plt.plot(fpr, tpr, color="gainsboro", alpha=1, linewidth=0.2)

    #     fpr, tpr, threshold = roc_curve(db.labels, db.predictions)
    #     plt.plot(fpr, tpr, color=COLORS[2])
    #     plt.plot([0, 1], [0, 1], color=COLORS[7], linewidth=0.5)

    #     plt.ylim(0, 1)
    #     plt.xlim(0, 1)
    #     # axis labels
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.show()

    # def roc_comp(self, fig_path="", ci=95, bootstrap_samples=1000):
    #     plt.figure(figsize=(6, 6), dpi=300)
    #     db0 = self.dataset_test_unique
    #     db1 = self.dataset_test_unique_subset_older50

    #     label_pred0 = np.stack((db0.labels, db0.predictions), axis=1)
    #     label_pred1 = np.stack((db1.labels, db1.predictions), axis=1)
    #     b0 = bootstrap(label_pred0, bootstrap_samples)
    #     b1 = bootstrap(label_pred1, bootstrap_samples)

    #     idx_sorted0 = rankdata(
    #         [roc_auc_score(x[:, 0], x[:, 1]) for x in b0], method="ordinal"
    #     )
    #     idx_sorted1 = rankdata(
    #         [roc_auc_score(x[:, 0], x[:, 1]) for x in b1], method="ordinal"
    #     )

    #     idx_min = int(0.05 * bootstrap_samples)
    #     idx_max = int(0.95 * bootstrap_samples)

    #     for i in range(len(idx_sorted0)):
    #         if idx_sorted0[i] >= idx_min and idx_sorted0[i] <= idx_max:
    #             fpr, tpr, _ = roc_curve(b0[i][:, 0], b0[i][:, 1])
    #             plt.plot(fpr, tpr, color="palegreen", alpha=1, linewidth=0.2)
    #             # plt.plot(fpr, tpr, color=COLORS[2], alpha=.5, linewidth=.2)

    #         if idx_sorted1[i] >= idx_min and idx_sorted1[i] <= idx_max:
    #             fpr, tpr, _ = roc_curve(b1[i][:, 0], b1[i][:, 1])
    #             plt.plot(fpr, tpr, color="lightskyblue", alpha=1, linewidth=0.2)
    #             # plt.plot(fpr, tpr, color=COLORS[0], alpha=.5, linewidth=.2)

    #     fpr, tpr, _ = roc_curve(db0.labels, db0.predictions)
    #     plt.plot(fpr, tpr, color=COLORS[2])
    #     fpr, tpr, _ = roc_curve(db1.labels, db1.predictions)
    #     plt.plot(fpr, tpr, color=COLORS[0])

    #     plt.plot([0, 1], [0, 1], color=COLORS[7], linewidth=0.5)

    #     plt.ylim(0, 1)
    #     plt.xlim(0, 1)
    #     # axis labels
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")

    #     if fig_path != "":
    #         plt.tight_layout(pad=1)
    #         plt.savefig(fig_path)

    #     plt.show()

    # def precision_recall_ci(self, ci=95, bootstrap_samples=1000):
    #     self.precision_recall_CI(self.dataset_test_unique, ci, bootstrap_samples)

    # def precision_recall_ci_older50(self, ci=95, bootstrap_samples=1000):
    #     self.precision_recall_CI(
    #         self.dataset_test_unique_subset_older50, ci, bootstrap_samples
    #     )

    # @staticmethod
    # def precision_recall_CI(db, ci, bootstrap_samples):
    #     plt.figure(figsize=(6, 6), dpi=300)
    #     # db = self.dataset_test_unique

    #     label_pred = np.stack((db.labels, db.predictions), axis=1)
    #     b = bootstrap(label_pred, bootstrap_samples)
    #     rcs = [average_precision_score(x[:, 0], x[:, 1]) for x in b]
    #     idx_sorted = rankdata(rcs, method="ordinal")

    #     idx_min = int(0.05 * bootstrap_samples)
    #     idx_max = int(0.95 * bootstrap_samples)

    #     for i in range(len(idx_sorted)):
    #         if idx_sorted[i] >= idx_min and idx_sorted[i] <= idx_max:
    #             precision, recall, _ = precision_recall_curve(b[i][:, 0], b[i][:, 1])
    #             plt.step(
    #                 recall,
    #                 precision,
    #                 where="post",
    #                 color="gainsboro",
    #                 alpha=1,
    #                 linewidth=0.2,
    #             )
    #             # plt.plot(fpr, tpr, color='gainsboro', alpha=1, linewidth=.2)

    #     precision, recall, _ = precision_recall_curve(db.labels, db.predictions)
    #     plt.step(recall, precision, where="post", color=COLORS[2])
    #     # plt.plot(fpr, tpr, color=COLORS[2])
    #     # plt.plot([0,1],[0,1], color=COLORS[7], linewidth=.5)

    #     plt.ylim(0, 1)
    #     plt.xlim(0, 1)
    #     # axis labels
    #     plt.xlabel("Recall")
    #     plt.ylabel("Precision")
    #     plt.show()

    # def precision_recall_comp(self, fig_path="", ci=95, bootstrap_samples=1000):
    #     plt.figure(figsize=(6, 6), dpi=300)

    #     db0 = self.dataset_test_unique
    #     db1 = self.dataset_test_unique_subset_older50

    #     label_pred0 = np.stack((db0.labels, db0.predictions), axis=1)
    #     label_pred1 = np.stack((db1.labels, db1.predictions), axis=1)

    #     b0 = bootstrap(label_pred0, bootstrap_samples)
    #     b1 = bootstrap(label_pred1, bootstrap_samples)
    #     idx_sorted0 = rankdata(
    #         [average_precision_score(x[:, 0], x[:, 1]) for x in b0], method="ordinal"
    #     )
    #     idx_sorted1 = rankdata(
    #         [average_precision_score(x[:, 0], x[:, 1]) for x in b0], method="ordinal"
    #     )

    #     idx_min = int(0.05 * bootstrap_samples)
    #     idx_max = int(0.95 * bootstrap_samples)

    #     for i in range(len(idx_sorted0)):
    #         if idx_sorted0[i] >= idx_min and idx_sorted0[i] <= idx_max:
    #             precision, recall, _ = precision_recall_curve(b0[i][:, 0], b0[i][:, 1])
    #             plt.step(
    #                 recall,
    #                 precision,
    #                 where="post",
    #                 color="palegreen",
    #                 alpha=0.2,
    #                 linewidth=0.5,
    #             )
    #         if idx_sorted1[i] >= idx_min and idx_sorted1[i] <= idx_max:
    #             precision, recall, _ = precision_recall_curve(b1[i][:, 0], b1[i][:, 1])
    #             plt.step(
    #                 recall,
    #                 precision,
    #                 where="post",
    #                 color="lightskyblue",
    #                 alpha=0.2,
    #                 linewidth=0.5,
    #             )

    #     precision, recall, _ = precision_recall_curve(db0.labels, db0.predictions)
    #     plt.step(recall, precision, where="post", color=COLORS[2])

    #     precision, recall, _ = precision_recall_curve(db1.labels, db1.predictions)
    #     plt.step(recall, precision, where="post", color=COLORS[0])

    #     plt.ylim(0, 1)
    #     plt.xlim(0, 1)
    #     # axis labels
    #     plt.xlabel("Recall")
    #     plt.ylabel("Precision")

    #     if fig_path != "":
    #         plt.tight_layout(pad=1)
    #         plt.savefig(fig_path)

    #     plt.show()

    # def precision_recall_curves(self):
    #     db = self.dataset_test_unique
    #     for _, pr in db.predictions_per_age.items():
    #         precision, recall, _ = precision_recall_curve(db.labels, pr)
    #         plt.step(recall, precision, where="post")
    #     plt.xlabel("Recall")
    #     plt.ylabel("Precision")
    #     plt.ylim([0.0, 1.0])
    #     plt.xlim([0.0, 1.0])
    #     plt.show()

    # Reports for subset older 50
    # def auc_older50(self):
    #     db = self.dataset_test_unique_subset_older50
    #     auc = roc_auc_score(db.labels, db.predictions)
    #     print("AUC = {:.3f}".format(auc))

    # def f1_older50(self):
    #     db = self.dataset_test_unique_subset_older50
    #     f1 = f1_score(db.labels, np.round(db.predictions))
    #     print("F1-Score = {:.3f}".format(f1))

    # def acc_older50(self):
    #     db = self.dataset_test_unique_subset_older50
    #     acc = accuracy_score(db.labels, np.round(db.predictions))
    #     print("Accuracy = {:.3f}".format(acc))

    # def bacc_older50(self):
    #     db = self.dataset_test_unique_subset_older50
    #     bacc = balanced_accuracy_score(db.labels, np.round(db.predictions))
    #     print("Balanced Accuracy = {:.3f}".format(bacc))

    # def auc_per_age_older50(self):
    #     db = self.dataset_test_unique_subset_older50
    #     aucs = {
    #         age: roc_auc_score(db.labels, preds)
    #         for age, preds in db.predictions_per_age.items()
    #     }
    #     for k, v in aucs.items():
    #         print("Predictions using patient age = {} have AUC = {:.3f}".format(k, v))

    # def f1_per_age_older50(self):
    #     db = self.dataset_test_unique_subset_older50
    #     f1s = {
    #         age: f1_score(db.labels, np.round(preds))
    #         for age, preds in db.predictions_per_age.items()
    #     }
    #     for k, v in f1s.items():
    #         print(
    #             "Predictions using patient age = {} have F1-Score = {:.3f}".format(k, v)
    #         )

    # def acc_per_age_older50(self):
    #     db = self.dataset_test_unique_subset_older50
    #     accs = {
    #         age: accuracy_score(db.labels, np.round(preds))
    #         for age, preds in db.predictions_per_age.items()
    #     }
    #     for k, v in accs.items():
    #         print(
    #             "Predictions using patient age = {} have Accuracy = {:.3f}".format(k, v)
    #         )

    # def bacc_per_age_older50(self):
    #     db = self.dataset_test_unique_subset_older50
    #     baccs = {
    #         age: balanced_accuracy_score(db.labels, np.round(preds))
    #         for age, preds in db.predictions_per_age.items()
    #     }
    #     for k, v in baccs.items():
    #         print(
    #             "Predictions using patient age = {} have Balanced Accuracy = {:.3f}".format(
    #                 k, v
    #             )
    #         )

    # def roc_curves_older50(self):
    #     db = self.dataset_test_unique_subset_older50
    #     for _, pr in db.predictions_per_age.items():
    #         fpr, tpr, threshold = roc_curve(db.labels, pr)
    #         plt.plot(fpr, tpr)
    #     plt.ylim(0, 1)
    #     plt.xlim(0, 1)
    #     # axis labels
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.show()

    # def precision_recall_curves_older50(self):
    #     db = self.dataset_test_unique_subset_older50
    #     for _, pr in db.predictions_per_age.items():
    #         precision, recall, _ = precision_recall_curve(db.labels, pr)
    #         plt.step(recall, precision, where="post")
    #     plt.xlabel("Recall")
    #     plt.ylabel("Precision")
    #     plt.ylim([0.0, 1.0])
    #     plt.xlim([0.0, 1.0])
    #     plt.show()

    # family test
    def family_plots(self, fig_path=""):
        db = self.dataset_test_unique
        df_pedigree = pd.read_csv("../data/datasets/pedigree.csv")
        df_pedigree = df_pedigree.set_index("id")

        ids = db.df[["id", "AGE", "sex", "mo", "fa", "mo_t2d", "fa_t2d", "T2D"]]
        ids = ids.join(df_pedigree[["famid"]], how="left", on="id")

        color_by_family = ids["famid"].values
        famid_sorted = np.unique(np.sort(color_by_family))
        fam_masks = [color_by_family == i for i in famid_sorted]
        patient_ages = np.array(db.features[:, db._feat_names.index("AGE")])

        plt.figure(figsize=(16, 32), dpi=150)
        for i in range(len(fam_masks)):
            ax1 = plt.subplot(8, 4, i + 1)
            p = np.stack(
                [db.predictions_per_age[age][fam_masks[i]] for age in range(20, 80, 5)],
                axis=0,
            )
            m_p = db.labels[fam_masks[i]] == 1
            m_n = db.labels[fam_masks[i]] == 0

            pa = patient_ages[fam_masks[i]]
            pp = db.predictions[fam_masks[i]]

            if True in m_n:
                plt.plot(p[:, m_n], color="grey")
            if True in m_p:
                plt.plot(p[:, m_p], color="red")
            # print patient age
            plt.scatter((pa[m_p] - 20) / 5, pp[m_p], c="black", marker="^")
            plt.scatter((pa[m_n] - 20) / 5, pp[m_n], c="blue", marker="+")

            plt.title("famid {}".format(famid_sorted[i]))
            plt.xticks(range(12), np.arange(20, 80, 5))
            plt.ylim(0, 1)
            # break

        if fig_path != "":
            plt.tight_layout(pad=1)
            plt.savefig(fig_path)

        plt.show()

    def parents_plots(self, fig_path=""):
        db = self.dataset_test_unique
        df = db.df.set_index("id")
        # print(df)
        df_fathers = df[df.index.isin(df.fa)]
        df_mothers = df[df.index.isin(df.mo)]
        df_children = df[
            (df.fa.isin(df_fathers.index)) & (df.mo.isin(df_mothers.index))
        ]
        # print(df_children)
        parents = df_children[["fa", "mo"]].groupby(["fa", "mo"]).count().index
        predictions = np.stack(
            [db.predictions_per_age[age] for age in range(20, 80, 5)], axis=1
        )
        patient_ages = np.array(db.features[:, db._feat_names.index("AGE")])
        # print(patient_ages)

        plt.figure(figsize=(16, 36), dpi=150)
        for i in range(len(parents)):
            ax1 = plt.subplot(9, 4, i + 1)
            plt.title("father {} mother {}".format(parents[i][0], parents[i][1]))
            father_mask = (db.df["id"] == parents[i][0]).values
            mother_mask = (db.df["id"] == parents[i][1]).values
            pos_mask = db.labels == 1
            neg_mask = db.labels == 0
            children_mask = np.logical_and(
                (db.df["fa"] == parents[i][0]).values,
                (db.df["mo"] == parents[i][1]).values,
            )

            m_pos_fa = np.logical_and(pos_mask, father_mask)
            m_pos_mo = np.logical_and(pos_mask, mother_mask)
            m_pos_chi = np.logical_and(pos_mask, children_mask)
            m_neg_fa = np.logical_and(neg_mask, father_mask)
            m_neg_mo = np.logical_and(neg_mask, mother_mask)
            m_neg_chi = np.logical_and(neg_mask, children_mask)

            if True in m_pos_fa:
                plt.plot(np.transpose(predictions[m_pos_fa, :]), color="blue")
            if True in m_pos_mo:
                plt.plot(np.transpose(predictions[m_pos_mo, :]), color="red")
            if True in m_pos_chi:
                plt.plot(np.transpose(predictions[m_pos_chi, :]), color="green")
            if True in m_neg_fa:
                plt.plot(np.transpose(predictions[m_neg_fa, :]), color="blue", ls="--")
            if True in m_neg_mo:
                plt.plot(np.transpose(predictions[m_neg_mo, :]), color="red", ls="--")
            if True in m_neg_chi:
                plt.plot(
                    np.transpose(predictions[m_neg_chi, :]), color="green", ls="--"
                )
            plt.scatter(
                (patient_ages[father_mask] - 20) / 5,
                db.predictions[father_mask],
                c="blue",
            )
            plt.scatter(
                (patient_ages[mother_mask] - 20) / 5,
                db.predictions[mother_mask],
                c="red",
            )
            plt.scatter(
                (patient_ages[children_mask] - 20) / 5,
                db.predictions[children_mask],
                c="green",
            )
            plt.ylim(0, 1)
            plt.xticks(range(12), np.arange(20, 80, 5))

        if fig_path != "":
            plt.tight_layout(pad=1)
            plt.savefig(fig_path)

        plt.show()

    # def first_positives(self):
    #     db = self.dataset_test_first_diag
    #     pos_mask = db.labels == 1
    #     pred = np.stack([db.predictions_per_age[age][pos_mask] for age in range(20,80,5)], axis=0)
    #     age_above_50 = db.df.AGE.values[pos_mask] >= 50
    #     age_below_50 = db.df.AGE.values[pos_mask] < 50

    #     neg = self.negatives_older60
    #     pred_below_50 = [np.array(x) for x in pred[:,age_below_50]]
    #     pred_above_50 = [np.array(x) for x in pred[:,age_above_50]]

    #     plt.figure(figsize=(15,5))
    #     plt.subplot(131)
    #     sns.boxplot(x=[i for i in neg[1]], y=pred_below_50, showfliers=False)
    #     plt.xlabel("positives\n(age < 50 )")
    #     plt.ylim(0,1)
    #     plt.subplot(132)
    #     sns.boxplot(x=[i for i in neg[1]], y=pred_above_50, showfliers=False)
    #     plt.xlabel("positives\n(age >= 50 )")
    #     plt.ylim(0,1)
    #     plt.subplot(133)
    #     sns.boxplot(x=[i for i in neg[1]], y=neg[0], showfliers=False)
    #     plt.xlabel("negatives\n(age >= 60)")
    #     plt.ylim(0,1)

    #     plt.show()
    # def plot_metrics(self, bootnum=1000, interval="HDI"):
    #     metrics = OrderedDict({
    #         'AUC': [self.auc, self.auc_neg_older50],
    #         'AvgPrec': [self.average_precision, self.average_precision_neg_older50],
    #         'BACC': [self.bacc, self.bacc_neg_older50],
    #         # 'ACC': [self.acc, self.acc_neg_older50],
    #         'F1': [self.f1, self.f1_neg_older50],
    #         'Precision': [self.precision, self.precision_neg_older50],
    #         'Sensitivity': [self.sensitivity, self.sensitivity_neg_older50],
    #         'Specificity': [self.specificity, self.specificity_neg_older50],
    #         'Brier': [self.brier, self.brier_neg_older50],
    #         'ECE': [self.ece, self.ece_neg_older50],
    #         'MCE': [self.mce, self.mce_neg_older50],
    #         })
    #     co = sns.color_palette("coolwarm", n_colors=33)
    #     fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, dpi=300)
    #     for (i,k) in enumerate(metrics.keys()):
    #         v = metrics[k][0](bootnum=bootnum, interval=interval)
    #         ax0.errorbar(v["value"], i, xerr=[[v["value"]-v["lower"]], [v["upper"]-v["value"]]], fmt='.', capsize=3.0, color=co[int(v["value"]*33)]) 
    #         vo = metrics[k][1](bootnum=bootnum, interval=interval)
    #         ax1.errorbar(vo["value"], i, xerr=[[vo["value"]-vo["lower"]], [vo["upper"]-vo["value"]]], fmt='.', capsize=3.0, color=co[int(vo["value"]*33)]) 

    #     ax0.set_xticks(np.arange(0,1.01,.2))
    #     ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    #     ax1.xaxis.grid(True, which='minor')
    #     ax1.set_xticks(np.arange(0,1.01,.2))
    #     ax0.xaxis.set_minor_locator(AutoMinorLocator(2))
    #     ax0.xaxis.grid(True, which='minor')
    #     ax0.set_ylim(len(metrics), -1)
    #     ax0.set_yticks(range(len(metrics)))
    #     ax0.set_yticklabels(metrics.keys())
    #     plt.show()

    # def first_positives(self, fig_path=""):
    #     db = self.dataset_test_first_diag
    #     pos_mask = db.labels == 1
    #     pred = np.stack(
    #         [db.predictions_per_age[age][pos_mask] for age in range(20, 80, 5)], axis=0
    #     )
    #     age_above_50 = db.df.AGE.values[pos_mask] >= 50
    #     age_below_50 = db.df.AGE.values[pos_mask] < 50

    #     neg = self.negatives_older60
    #     pred_below_50 = [np.array(x) for x in pred[:, age_below_50]]
    #     pred_below_50 = [x for x in pred[:, age_below_50]]
    #     pred_above_50 = [np.array(x) for x in pred[:, age_above_50]]
    #     # print(pred_below_50)
    #     # x = [i for i in neg[1]]
    #     # print(x)

    #     color_boxplot = sns.color_palette("cool", n_colors=20)

    #     plt.figure(figsize=(15, 5), dpi=300)
    #     plt.subplot(131)
    #     bp0 = plt.boxplot(
    #         pred_below_50,
    #         showfliers=False,
    #         patch_artist=True,
    #         labels=[i for i in neg[1]],
    #         medianprops=dict(linewidth=2.5, color="black"),
    #     )
    #     colors = [color_boxplot[int(np.median(a) * 20)] for a in pred_below_50]
    #     for box, color in zip(bp0["boxes"], colors):
    #         box.set(facecolor=color)
    #         # box.set(color='blue', linewidth=5)
    #         # box.set_facecolor('red')
    #     # sns.boxplot(x=[i for i in neg[1]], y=pred_below_50, showfliers=False, palette=[color_boxplot[int(np.median(a)*20)] for a in pred_below_50])
    #     # sns.boxplot(x=np.array([i for i in neg[1]]), y=np.array(pred_below_50))
    #     # sns.boxplot(y=pred_below_50[0])
    #     # sns.catplot(x=[0,1], y=pred_below_50[0:2], kind='box')
    #     # plt.boxplot(pred_below_50)

    #     plt.xlabel("positives\n(age < 50 )")
    #     plt.ylim(0, 1)
    #     plt.subplot(132)
    #     bp1 = plt.boxplot(
    #         pred_above_50,
    #         showfliers=False,
    #         patch_artist=True,
    #         labels=[i for i in neg[1]],
    #         medianprops=dict(linewidth=2.5, color="black"),
    #     )
    #     colors = [color_boxplot[int(np.median(a) * 20)] for a in pred_above_50]
    #     for box, color in zip(bp1["boxes"], colors):
    #         box.set(facecolor=color)
    #     # sns.boxplot(x=[i for i in neg[1]], y=pred_above_50, showfliers=False, palette=[color_boxplot[int(np.median(a)*20)] for a in pred_above_50])
    #     plt.xlabel("positives\n(age >= 50 )")
    #     plt.ylim(0, 1)
    #     plt.subplot(133)
    #     bp2 = plt.boxplot(
    #         neg[0],
    #         showfliers=False,
    #         patch_artist=True,
    #         labels=[i for i in neg[1]],
    #         medianprops=dict(linewidth=2.5, color="black"),
    #     )
    #     colors = [color_boxplot[int(np.median(a) * 20)] for a in neg[0]]
    #     for box, color in zip(bp2["boxes"], colors):
    #         box.set(facecolor=color)
    #         box.set(edgecolor="black")
    #     # sns.boxplot(x=[i for i in neg[1]], y=neg[0], showfliers=False, palette=[color_boxplot[int(np.median(a)*20)] for a in neg[0]])
    #     plt.xlabel("negatives\n(age >= 60)")
    #     plt.ylim(0, 1)
    #     plt.tight_layout(pad=1)

    #     if fig_path != "":
    #         plt.savefig(fig_path)

    #     plt.show()

    # def first_positives_violin(self, fig_path=""):
    #     db = self.dataset_test_first_diag
    #     pos_mask = db.labels == 1
    #     pred = np.stack([db.predictions_per_age[age][pos_mask] for age in range(20,80,5)], axis=0)
    #     age_above_50 = db.df.AGE.values[pos_mask] >= 50
    #     age_below_50 = db.df.AGE.values[pos_mask] < 50

    #     neg = self.negatives_older60
    #     pred_below_50 = [np.array(x) for x in pred[:,age_below_50]]
    #     pred_above_50 = [np.array(x) for x in pred[:,age_above_50]]

    #     color_boxplot = sns.color_palette("cool", n_colors=20)

    #     plt.figure(figsize=(15,5))
    #     plt.subplot(131)
    #     sns.violinplot(x=[i for i in neg[1]], y=pred_below_50, showfliers=False, palette=[color_boxplot[int(np.median(a)*20)] for a in pred_below_50])
    #     plt.xlabel("positives\n(age < 50 )")
    #     plt.ylim(0,1)
    #     plt.subplot(132)
    #     sns.violinplot(x=[i for i in neg[1]], y=pred_above_50, showfliers=False, palette=[color_boxplot[int(np.median(a)*20)] for a in pred_above_50])
    #     plt.xlabel("positives\n(age >= 50 )")
    #     plt.ylim(0,1)
    #     plt.subplot(133)
    #     sns.violinplot(x=[i for i in neg[1]], y=neg[0], showfliers=False, palette=[color_boxplot[int(np.median(a)*20)] for a in neg[0]])
    #     plt.xlabel("negatives\n(age >= 60)")
    #     plt.ylim(0,1)
    #     plt.tight_layout(pad=1)

    #     if fig_path != "":
    #         plt.savefig(fig_path)

    #     plt.show()

    # def first_positives_boxen(self, fig_path=""):
    #     db = self.dataset_test_first_diag
    #     pos_mask = db.labels == 1
    #     pred = np.stack([db.predictions_per_age[age][pos_mask] for age in range(20,80,5)], axis=0)
    #     age_above_50 = db.df.AGE.values[pos_mask] >= 50
    #     age_below_50 = db.df.AGE.values[pos_mask] < 50

    #     neg = self.negatives_older60
    #     pred_below_50 = [np.array(x) for x in pred[:,age_below_50]]
    #     pred_above_50 = [np.array(x) for x in pred[:,age_above_50]]

    #     color_boxplot = sns.color_palette("cool", n_colors=20)

    #     plt.figure(figsize=(15,5))
    #     plt.subplot(131)
    #     sns.boxenplot(x=[i for i in neg[1]], y=pred_below_50, palette=[color_boxplot[int(np.median(a)*20)] for a in pred_below_50])
    #     plt.xlabel("positives\n(age < 50 )")
    #     plt.ylim(0,1)
    #     plt.subplot(132)
    #     sns.boxenplot(x=[i for i in neg[1]], y=pred_above_50, palette=[color_boxplot[int(np.median(a)*20)] for a in pred_above_50])
    #     plt.xlabel("positives\n(age >= 50 )")
    #     plt.ylim(0,1)
    #     plt.subplot(133)
    #     sns.boxenplot(x=[i for i in neg[1]], y=neg[0], palette=[color_boxplot[int(np.median(a)*20)] for a in neg[0]])
    #     plt.xlabel("negatives\n(age >= 60)")
    #     plt.ylim(0,1)
    #     plt.tight_layout(pad=1)

    #     if fig_path != "":
    #         plt.savefig(fig_path)

    #     plt.show()

    # def use_first_diag(self):
    #     db = self.dataset_test_first_diag
    #     pos_mask = db.labels == 1
    #     neg_mask = db.labels == 0
    #     print(db.predictions[neg_mask],db.predictions[pos_mask])
    #     print((db.predictions[pos_mask]-db.predictions[neg_mask])*100)
    #     print(np.mean((db.predictions[pos_mask]-db.predictions[neg_mask]))*100)

    # def years_before_diag(self, age_at_prediction=20):
    #     db = self.dataset_test_first_diag
    #     pos_mask = db.labels == 1
    #     pred = db.predictions_per_age[age_at_prediction][pos_mask]
    #     age = db.df.AGE.values[pos_mask]-age_at_prediction
    #     # print(pred)
    #     # print(age)
    #     plt.scatter(pred,age)
    #     coef = np.polyfit(pred,age,1)
    #     poly1d_fn = np.poly1d(coef)
    #     print(coef)
    #     plt.plot(pred, age, 'ko', pred, poly1d_fn(pred), 'r')
    #     plt.xlim(0,1)
    #     plt.xlabel("Predicted probability (30yo)")
    #     plt.ylabel("Patient age at first positive diagnostic")
    #     plt.show()

    # def years_before_diag2(self):
    #     db = self.dataset_test_first_diag
    #     pos_mask = db.labels == 1
    #     pred = np.stack([db.predictions_per_age[age][pos_mask] for age in range(20,80,5)], axis=0)
    #     age_above_50 = db.df.AGE.values[pos_mask] > 50
    #     age_below_50 = db.df.AGE.values[pos_mask] < 50

    #     pred_above50_mean = np.mean(pred[:,age_above_50], axis=1)
    #     pred_above50_std = np.mean(pred[:,age_above_50], axis=1)
    #     pred_below50_mean = np.mean(pred[:,age_below_50], axis=1)
    #     pred_below50_std = np.std(pred[:,age_below_50], axis=1)

    #     neg = self.negatives_older60
    #     neg_mean = np.mean(np.array(neg[0]), axis=1)
    #     # neg_median = np.median(np.array(neg[0]), axis=1)
    #     neg_std = np.std(np.array(neg[0]), axis=1)

    #     # # age_inter =  np.logical_and((db.df.AGE.values[pos_mask] > 30), (db.df.AGE.values[pos_mask] < 70))
    #     # plt.plot(pred_above50_mean, color='green')
    #     plt.errorbar(x=np.arange(20.2,80.2,5), y=pred_above50_mean, yerr=pred_above50_std, color='green')
    #     # # plt.plot(np.mean(pred[:,age_inter], axis=1), color='yellow')
    #     # plt.plot(pred_below50_mean, color='red')
    #     plt.errorbar(x=np.arange(20.1,80.1,5), y=pred_below50_mean, yerr=pred_below50_std, color='red')
    #     # plt.plot(neg_mean, color='grey')
    #     plt.errorbar(x=np.arange(20,80,5), y=neg_mean, yerr=neg_std, color='grey')
    #     # plt.plot(neg_median, color='purple')
    #     # # plt.scatter(age, db.predictions[pos_mask])
    #     plt.show()


# if __name__ == "__main__":
#     report = DiabNetReport(
#         "../diabnet/models/teste-sp-soft-label_tmp.pth",
#         "positivo_1000_random_0.csv",
#         "../data/datasets/visits_sp_unique_test_positivo_1000_random_0_negatives_older60.csv",
#     )
#     report.first_positives2()
    # report.auc_per_age()
    # report.f1_per_age()
    # report.acc_per_age()
    # report.bacc_per_age()
    # report.roc_curves()

    # report.precision_recall_curves()
    # # subset older 60
    # report.auc_per_age_older50()
    # report.f1_per_age_older50()
    # report.acc_per_age_older50()
    # report.bacc_per_age_older50()
    # report.roc_curves_older50()
    # report.precision_recall_curves_older50()

    # print(report.dataset_test_changed.df)
    # report.family_plots()
    # report.parents_plots()
    # report.family_plots()

    # print(report.feat_names)
    # print(report.dataset_test_unique.features)
    # print(report.dataset_test_unique.labels)
    # print(report.dataset_test_unique.predictions)

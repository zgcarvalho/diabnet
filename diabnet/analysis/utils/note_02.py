import torch
import seaborn
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.utils.validation import indexable
from captum.attr import IntegratedGradients
from diabnet.data import encode_features
import matplotlib as mpl
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    # f1_score,
    # accuracy_score,
    # balanced_accuracy_score,
    # brier_score_loss,
    average_precision_score,
    # precision_score,
    # recall_score,  # if pos_label = (sensitivity or TPR) if neg_label = (specificity or TNR)
)
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from sklearn.calibration import calibration_curve
import matplotlib.patches as mpatches

COLORS = sns.color_palette("colorblind")

def plot_metrics(r, interval="HDI"):
    """Plot testing metrics.
    Parameters
    ----------
    r: DiabNetReport

    Returns
    -------
    figure
    """
    metrics = OrderedDict(
        {
            "AUROC": [r.auc, r.auc_neg_older50],
            "AUPRC": [r.average_precision, r.average_precision_neg_older50],
            "BACC": [r.bacc, r.bacc_neg_older50],
            # 'ACC': [r.acc, r.acc_neg_older50],
            "F1": [r.f1, r.f1_neg_older50],
            "Precision": [r.precision, r.precision_neg_older50],
            "Sensitivity": [r.sensitivity, r.sensitivity_neg_older50],
            "Specificity": [r.specificity, r.specificity_neg_older50],
            "Brier": [r.brier, r.brier_neg_older50],
            "ECE": [r.ece, r.ece_neg_older50],
            "MCE": [r.mce, r.mce_neg_older50],
        }
    )
    co = sns.color_palette("coolwarm", n_colors=33)
    # fig = plt.figure(dpi=300)
    # ax0 = fig.add_subplot(221)
    # ax1 = fig.add_subplot(222)
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax0.set_title("Full test set", fontsize=15)
    ax1.set_title("Subset without\nnegatives younger than 50", fontsize=14)
    for (i, k) in enumerate(metrics.keys()):
        v = metrics[k][0](bootnum=1000, interval=interval)
        ax0.errorbar(
            v["value"],
            i,
            xerr=[[v["value"] - v["lower"]], [v["upper"] - v["value"]]],
            fmt=".",
            capsize=3.0,
            color=co[int(v["value"] * 33)],
        )
        vo = metrics[k][1](bootnum=1000, interval=interval)
        ax1.errorbar(
            vo["value"],
            i,
            xerr=[[vo["value"] - vo["lower"]], [vo["upper"] - vo["value"]]],
            fmt=".",
            capsize=3.0,
            color=co[int(vo["value"] * 33)],
        )

    ax0.set_xticks(np.arange(0, 1.01, 0.2))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.xaxis.grid(True, which="minor")
    ax1.set_xticks(np.arange(0, 1.01, 0.2))
    ax0.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax0.xaxis.grid(True, which="minor")
    ax0.set_ylim(len(metrics), -1)
    ax0.set_yticks(range(len(metrics)))
    ax0.set_yticklabels(metrics.keys())
    ax1.set_ylim(len(metrics), -1)
    ax1.set_yticks(range(len(metrics)))
    ax1.set_yticklabels("")

    conf_matrix = r.confusion()
    conf_matrix_50 = r.confusion_neg_older50()
    ax2.matshow(conf_matrix, cmap=plt.cm.Purples, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax2.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large')

    ax3.matshow(conf_matrix_50, cmap=plt.cm.Purples, alpha=0.3)
    for i in range(conf_matrix_50.shape[0]):
        for j in range(conf_matrix_50.shape[1]):
            ax3.text(x=j, y=i,s=conf_matrix_50[i, j], va='center', ha='center', size='large')

    ax2.set_xlabel("Predicted", fontsize=18)
    ax2.set_ylabel("Actual", fontsize=18)
    ax2.set_yticks(range(2))
    ax2.set_yticklabels(["N", "P"])
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(["N", "P"])
    ax2.xaxis.tick_bottom()
    ax2.grid(False)

    ax3.set_xlabel("Predicted", fontsize=18)
    ax3.set_ylabel("Actual", fontsize=18)
    ax3.set_yticks(range(2))
    ax3.set_yticklabels(["N", "P"])
    ax3.set_xticks(range(2))
    ax3.set_xticklabels(["N", "P"])
    ax3.xaxis.tick_bottom()
    ax3.grid(False)

    # print()
    # print(r.confusion_neg_older50())

    return fig

def _hdi_idx(data, p):
    """Highest Density Interval"""
    l = int(len(data) * p)
    diff, lower, upper = data[-1] - data[0], data[0], data[-1]
    for i in range(len(data) - l):
        new_diff = data[i + l - 1] - data[i]
        if diff > new_diff:
            diff = new_diff
            lower = data[i]
            upper = data[i + l - 1]
            lower_idx = i
            upper_idx = i + l - 1
    return (lower_idx, upper_idx)

def _roc_b(
    db, ax, num=1000, alpha=0.05, interval="CI", colors=[COLORS[2], "gainsboro"]
):

    # bootstrap with ensemble
    n_patients = len(db.labels)
    n_models = len(db.predictions_ensemble[0])

    aucs = np.empty(num)
    data = np.empty((num, 2, n_patients))

    for i in range(num):
        mask_preds = np.random.choice(n_patients, n_patients)
        mask_ensemble = np.random.choice(n_models, n_models)

        aucs[i] = roc_auc_score(
            db.labels[mask_preds],
            np.mean(db.predictions_ensemble[mask_preds][:, mask_ensemble], 1),
        )
        data[i, 0] = db.labels[mask_preds]
        data[i, 1] = np.mean(db.predictions_ensemble[mask_preds][:, mask_ensemble], 1)

    sorting_idx = np.argsort(aucs)

    if interval == "HDI":
        lower_idx, upper_idx = _hdi_idx(aucs[sorting_idx], 1 - alpha)
    else:
        # CDI
        lower_idx = int(num * alpha / 2)
        upper_idx = int(num * (1 - alpha / 2))

    idx = sorting_idx[lower_idx:upper_idx]

    for i in idx:
        fpr, tpr, _ = roc_curve(data[i, 0], data[i, 1])
        ax.plot(fpr, tpr, color=colors[1], alpha=0.5, linewidth=0.2)

    fpr, tpr, _ = roc_curve(db.labels, db.predictions)
    ax.plot(fpr, tpr, color=colors[0])
    ax.text(0.7, 0.08, f'AUROC: {np.mean(aucs):.3f}', fontsize=21, horizontalalignment='center')
    ax.text(0.7, 0.05, f'{int((1-alpha)*100)}% {interval}: [{aucs[idx[0]]:.3f}, {aucs[idx[-1]]:.3f}]', fontsize=13, horizontalalignment='center')


    ax.plot([0, 1], [0, 1], color=COLORS[7], linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

def _roc_b_comp(
    db1,
    db2,
    ax,
    num=1000,
    alpha=0.05,
    interval="CI",
    colors=[[COLORS[2], "palegreen"], [COLORS[0], "lightskyblue"]],
):

    # bootstrap with ensemble
    n_patients_1 = len(db1.labels)
    n_patients_2 = len(db2.labels)
    # n_models are the same for db1 and db2
    n_models = len(db1.predictions_ensemble[0])

    aucs_1 = np.empty(num)
    aucs_2 = np.empty(num)
    data_1 = np.empty((num, 2, n_patients_1))
    data_2 = np.empty((num, 2, n_patients_2))

    for i in range(num):
        mask_preds_1 = np.random.choice(n_patients_1, n_patients_1)
        mask_preds_2 = np.random.choice(n_patients_2, n_patients_2)
        mask_ensemble_1 = np.random.choice(n_models, n_models)
        mask_ensemble_2 = np.random.choice(n_models, n_models)

        aucs_1[i] = roc_auc_score(
            db1.labels[mask_preds_1],
            np.mean(db1.predictions_ensemble[mask_preds_1][:, mask_ensemble_1], 1),
        )
        aucs_2[i] = roc_auc_score(
            db2.labels[mask_preds_2],
            np.mean(db2.predictions_ensemble[mask_preds_2][:, mask_ensemble_2], 1),
        )
        data_1[i, 0] = db1.labels[mask_preds_1]
        data_2[i, 0] = db2.labels[mask_preds_2]
        data_1[i, 1] = np.mean(
            db1.predictions_ensemble[mask_preds_1][:, mask_ensemble_1], 1
        )
        data_2[i, 1] = np.mean(
            db2.predictions_ensemble[mask_preds_2][:, mask_ensemble_2], 1
        )

    sorting_idx_1 = np.argsort(aucs_1)
    sorting_idx_2 = np.argsort(aucs_2)

    if interval == "HDI":
        lower_idx_1, upper_idx_1 = _hdi_idx(aucs_1[sorting_idx_1], 1 - alpha)
        lower_idx_2, upper_idx_2 = _hdi_idx(aucs_2[sorting_idx_2], 1 - alpha)
    else:
        # CDI
        lower_idx_1 = int(num * alpha / 2)
        lower_idx_2 = lower_idx_1
        upper_idx_1 = int(num * (1 - alpha / 2))
        upper_idx_2 = upper_idx_1

    idx_1 = sorting_idx_1[lower_idx_1:upper_idx_1]
    idx_2 = sorting_idx_2[lower_idx_2:upper_idx_2]

    for i in range(len(idx_1)):
        fpr, tpr, _ = roc_curve(data_1[idx_1[i], 0], data_1[idx_1[i], 1])
        ax.plot(fpr, tpr, color=colors[0][1], alpha=0.3, linewidth=0.1)
        fpr, tpr, _ = roc_curve(data_2[idx_2[i], 0], data_2[idx_2[i], 1])
        ax.plot(fpr, tpr, color=colors[1][1], alpha=0.2, linewidth=0.1)

    fpr, tpr, _ = roc_curve(db1.labels, db1.predictions)
    ax.plot(fpr, tpr, color=colors[0][0])
    fpr, tpr, _ = roc_curve(db2.labels, db2.predictions)
    ax.plot(fpr, tpr, color=colors[1][0])

    ax.plot([0, 1], [0, 1], color=COLORS[7], linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

def _roc_by_ages(db, ax):
    colors = sns.color_palette("BrBG_r", n_colors=12)
    for age, preds in db.predictions_per_age.items():
        fpr, tpr, _ = roc_curve(db.labels, preds)
        score = roc_auc_score(db.labels, preds)
        ax.plot(fpr, tpr, color=colors[int((age-20)/5)],label=f"{age} : {score:.3f}")
        

    ax.legend(
        title='Age : AUROC',
        title_fontsize='large',
        loc="lower right",
        prop={'size': 13},
    )
    ax.plot([0, 1], [0, 1], color=COLORS[7], linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

def plot_roc_by_age(r):
    # fig = plt.figure(figsize=(6, 6), dpi=300)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    _roc_by_ages(r.dataset_test_unique, ax1)
    return fig

def plot_roc_by_age_neg_older50(r):
    # fig = plt.figure(figsize=(6, 6), dpi=300)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    _roc_by_ages(r.dataset_test_unique_subset_neg_older50, ax1)
    return fig

def plot_roc(r, interval="HDI"):
    # fig = plt.figure(figsize=(6, 6), dpi=300)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    c = [COLORS[2], "gainsboro"]
    _roc_b(
        r.dataset_test_unique, ax1, num=2000, alpha=0.05, interval=interval, colors=c
    )
    return fig

def plot_roc_neg_older50(r, interval="HDI"):
    # fig = plt.figure(figsize=(6, 6), dpi=300)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    c = [COLORS[0], "gainsboro"]
    _roc_b(
        r.dataset_test_unique_subset_neg_older50,
        ax1,
        num=2000,
        alpha=0.05,
        interval=interval,
        colors=c,
    )
    return fig

def plot_roc_comp(r, interval="HDI"):
    # fig = plt.figure(figsize=(6, 6), dpi=300)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    _roc_b_comp(
        r.dataset_test_unique,
        r.dataset_test_unique_subset_neg_older50,
        ax1,
        num=1000,
        alpha=0.05,
        interval=interval,
    )
    return fig

def _precision_recall_b(
    db, ax, num=1000, alpha=0.05, interval="CI", colors=[COLORS[2], "gainsboro"]
):

    # bootstrap with ensemble
    n_patients = len(db.labels)
    n_models = len(db.predictions_ensemble[0])

    avgps = np.empty(num)
    data = np.empty((num, 2, n_patients))

    for i in range(num):
        mask_preds = np.random.choice(n_patients, n_patients)
        mask_ensemble = np.random.choice(n_models, n_models)

        avgps[i] = average_precision_score(
            db.labels[mask_preds],
            np.mean(db.predictions_ensemble[mask_preds][:, mask_ensemble], 1),
        )
        data[i, 0] = db.labels[mask_preds]
        data[i, 1] = np.mean(db.predictions_ensemble[mask_preds][:, mask_ensemble], 1)

    sorting_idx = np.argsort(avgps)

    if interval == "HDI":
        lower_idx, upper_idx = _hdi_idx(avgps[sorting_idx], 1 - alpha)
    else:
        # CDI
        lower_idx = int(num * alpha / 2)
        upper_idx = int(num * (1 - alpha / 2))

    idx = sorting_idx[lower_idx:upper_idx]

    for i in idx:
        precision, recall, _ = precision_recall_curve(data[i, 0], data[i, 1])
        ax.step(
            recall,
            precision,
            where="post",
            color=colors[1],
            alpha=0.5,
            linewidth=0.2,
        )

    precision, recall, _ = precision_recall_curve(db.labels, db.predictions)
    ax.step(recall, precision, where="post", color=colors[0])
    ax.text(0.3, 0.08, f'AUPRC: {np.mean(avgps):.3f}', fontsize=21, horizontalalignment='center')
    ax.text(0.3, 0.05, f'{int((1-alpha)*100)}% {interval}: [{avgps[idx[0]]:.3f}, {avgps[idx[-1]]:.3f}]', fontsize=13, horizontalalignment='center')

    ax.plot([0, 1], [0, 1], color=COLORS[7], linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

def _precision_recall_b_comp(
    db1,
    db2,
    ax,
    num=1000,
    alpha=0.05,
    interval="CI",
    colors=[[COLORS[2], "palegreen"], [COLORS[0], "lightskyblue"]],
):

    # bootstrap with ensemble
    n_patients_1 = len(db1.labels)
    n_patients_2 = len(db2.labels)
    # n_models are the same for db1 and db2
    n_models = len(db1.predictions_ensemble[0])

    avgs_1 = np.empty(num)
    avgs_2 = np.empty(num)
    data_1 = np.empty((num, 2, n_patients_1))
    data_2 = np.empty((num, 2, n_patients_2))

    for i in range(num):
        mask_preds_1 = np.random.choice(n_patients_1, n_patients_1)
        mask_preds_2 = np.random.choice(n_patients_2, n_patients_2)
        mask_ensemble_1 = np.random.choice(n_models, n_models)
        mask_ensemble_2 = np.random.choice(n_models, n_models)

        avgs_1[i] = average_precision_score(
            db1.labels[mask_preds_1],
            np.mean(db1.predictions_ensemble[mask_preds_1][:, mask_ensemble_1], 1),
        )
        avgs_2[i] = average_precision_score(
            db2.labels[mask_preds_2],
            np.mean(db2.predictions_ensemble[mask_preds_2][:, mask_ensemble_2], 1),
        )
        data_1[i, 0] = db1.labels[mask_preds_1]
        data_2[i, 0] = db2.labels[mask_preds_2]
        data_1[i, 1] = np.mean(
            db1.predictions_ensemble[mask_preds_1][:, mask_ensemble_1], 1
        )
        data_2[i, 1] = np.mean(
            db2.predictions_ensemble[mask_preds_2][:, mask_ensemble_2], 1
        )

    sorting_idx_1 = np.argsort(avgs_1)
    sorting_idx_2 = np.argsort(avgs_2)

    if interval == "HDI":
        lower_idx_1, upper_idx_1 = _hdi_idx(avgs_1[sorting_idx_1], 1 - alpha)
        lower_idx_2, upper_idx_2 = _hdi_idx(avgs_2[sorting_idx_2], 1 - alpha)
    else:
        # CDI
        lower_idx_1 = int(num * alpha / 2)
        lower_idx_2 = lower_idx_1
        upper_idx_1 = int(num * (1 - alpha / 2))
        upper_idx_2 = upper_idx_1

    idx_1 = sorting_idx_1[lower_idx_1:upper_idx_1]
    idx_2 = sorting_idx_2[lower_idx_2:upper_idx_2]

    for i in range(len(idx_1)):
        precision, recall, _ = precision_recall_curve(
            data_1[idx_1[i], 0], data_1[idx_1[i], 1]
        )
        ax.step(
            recall,
            precision,
            where="post",
            color=colors[0][1],
            alpha=0.2,
            linewidth=0.1,
        )
        precision, recall, _ = precision_recall_curve(
            data_2[idx_2[i], 0], data_2[idx_2[i], 1]
        )
        ax.step(
            recall,
            precision,
            where="post",
            color=colors[1][1],
            alpha=0.3,
            linewidth=0.1,
        )

    precision, recall, _ = precision_recall_curve(db1.labels, db1.predictions)
    ax.step(recall, precision, where="post", color=colors[0][0])
    precision, recall, _ = precision_recall_curve(db2.labels, db2.predictions)
    ax.step(recall, precision, where="post", color=colors[1][0])

    ax.plot([0, 1], [0, 1], color=COLORS[7], linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

def _precision_recall_by_ages(db, ax):
    colors = sns.color_palette("BrBG_r", n_colors=12)
    for age, preds in db.predictions_per_age.items():
        precision, recall, _ = precision_recall_curve(db.labels, preds)
        score = average_precision_score(db.labels, preds)
        ax.step(
            recall,
            precision,
            where="post",
            color=colors[int((age-20)/5)],
            # alpha=0.5,
            # linewidth=0.2,
            label=f"{age} : {score:.3f}"
        )

    ax.legend(
        title='Age : AUPRC',
        title_fontsize='large',
        loc="lower left",
        prop={'size': 13},
    )
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

def plot_precision_recall_by_age(r):
    # fig = plt.figure(figsize=(6, 6), dpi=300)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    _precision_recall_by_ages(r.dataset_test_unique, ax1)
    return fig

def plot_precision_recall_by_age_neg_older50(r):
    # fig = plt.figure(figsize=(6, 6), dpi=300)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    _precision_recall_by_ages(r.dataset_test_unique_subset_neg_older50, ax1)
    return fig

def plot_precision_recall(r, interval="HDI"):
    # fig = plt.figure(figsize=(6, 6), dpi=300)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    c = [COLORS[2], "gainsboro"]
    _precision_recall_b(
        r.dataset_test_unique, ax1, num=2000, alpha=0.05, interval=interval, colors=c
    )
    return fig

def plot_precision_recall_neg_older50(r, interval="HDI"):
    # fig = plt.figure(figsize=(6, 6), dpi=300)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    c = [COLORS[0], "gainsboro"]
    _precision_recall_b(
        r.dataset_test_unique_subset_neg_older50,
        ax1,
        num=2000,
        alpha=0.05,
        interval=interval,
        colors=c,
    )
    return fig

def plot_precision_recall_comp(r, interval="HDI"):
    # fig = plt.figure(figsize=(6, 6), dpi=300)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    _precision_recall_b_comp(r.dataset_test_unique, r.dataset_test_unique_subset_neg_older50, ax1, num=1000, alpha=0.05, interval=interval)
    return fig

def _kde_plot(x,ax,color):
    if len(x) > 100:
        x = x[::100]
    x_d = np.linspace(0, 1, 1000)

    kde = KernelDensity(bandwidth=0.17, kernel='gaussian', metric='euclidean')
    kde.fit(x[:, None])

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])
    prob = np.exp(logprob)

    ax.plot(x_d, prob, color=color)

def plot_first_positives_lines(r, age_separator=(40,60)):
    db = r.dataset_test_first_diag
    pos_mask = db.labels == 1
    pred = np.stack([db.predictions_per_age[age][pos_mask] for age in range(20,80,5)], axis=0)
    age_below = db.df.AGE.values[pos_mask] < age_separator[0]  
    age_above = db.df.AGE.values[pos_mask] >= age_separator[1] 
    

    neg = r.negatives_older60
    pred_below = [np.array(x) for x in pred[:,age_below]]
    pred_below = [x for x in pred[:,age_below]]
    pred_above = [np.array(x) for x in pred[:,age_above]]

    # color_boxplot = sns.color_palette("copper_r", n_colors=14)
    color_boxplot = sns.color_palette("BrBG_r", n_colors=12)

    # bandwidth(pred_below_50[4])

    # fig, (ax0, ax1, ax2) = plt.figure(figsize=(15,5), dpi=300)
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, sharey=False, figsize=(18,6))
    # ax0 = plt.subplot(131)
    for i in range(12):
        kde0 = _kde_plot(pred_below[i], ax0, color=color_boxplot[i])
    ax0.set_title(f"Diabetics\n(first diagnostic before {age_separator[0]} years)")
    ax0.set_ylabel("Density", fontsize=15)
    ax0.set_xlabel("Risk score", fontsize=15)
    ax0.set_xlim(-0.01, 1.01)
    ax0.set_ylim(-0.0001, 2.5)

    # ax1.subplot(132)
    for i in range(12):
        kde1 = _kde_plot(pred_above[i], ax1, color=color_boxplot[i])
    ax1.set_title(f"Diabetics\n(first diagnostic after {age_separator[1]} years)")
    ax1.set_ylabel("Density", fontsize=15)
    ax1.set_xlabel("Risk score", fontsize=15)
    ax1.set_xlim(-0.01, 1.01)
    ax1.set_ylim(-0.0001, 2.5)
    
    # ax2.subplot(133)
    for i in range(12):
        kde2 = _kde_plot(neg[0][i], ax2, color=color_boxplot[i])
    ax2.set_title("Non-diabetics\n(with 60 years or more)")
    ax2.set_ylabel("Density", fontsize=15)
    ax2.set_xlabel("Risk score", fontsize=15)
    ax2.set_xlim(-0.01, 1.01)
    ax2.set_ylim(-0.0001, 2.5)
    fig.tight_layout(pad=1)
 
    cmap = mpl.colors.ListedColormap([color_boxplot[i] for i in range(12)])
    norm = mpl.colors.Normalize(vmin=17.5, vmax=77.5)
    cbax = fig.add_axes([1.0, 0.13, 0.02, 0.70]) 
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbax, orientation='vertical',ticks=np.arange(20,80,5))
    cbax.set_ylabel('Age (model input)', fontsize=15)

    return fig

def plot_first_positives_boxplot(r, age_separator=(40,60)):
    db = r.dataset_test_first_diag
    pos_mask = db.labels == 1
    pred = np.stack([db.predictions_per_age[age][pos_mask] for age in range(20,80,5)], axis=0)
    age_below = db.df.AGE.values[pos_mask] < age_separator[0]
    age_above = db.df.AGE.values[pos_mask] >= age_separator[1]
    

    neg = r.negatives_older60
    pred_below = [np.array(x) for x in pred[:,age_below]]
    pred_below = [x for x in pred[:,age_below]]
    pred_above = [np.array(x) for x in pred[:,age_above]]

    color_boxplot = sns.color_palette("Spectral_r", n_colors=20)
    # color_boxplot = sns.color_palette("cool", n_colors=20)

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, sharey=False, figsize=(18,6))
    # box 01
    ax0.boxplot(
            pred_below,
            showfliers=False,
            patch_artist=True,
            labels=[i for i in neg[1]],
            medianprops=dict(linewidth=2.5, color="black"),
    )
    colors = [color_boxplot[int(np.mean(a) * 20)] for a in pred_below]
    for (i,box) in enumerate(ax0.artists):
        box.set_facecolor(colors[i])
    # ax0.set_title("positives\n(age < 50 )")
    ax0.set_title(f"Diabetics\n(first diagnostic before {age_separator[0]} years)")
    ax0.set_ylabel("Risk score", fontsize=15)
    ax0.set_xlabel("Age (model input)", fontsize=15)
    ax0.set_ylim(0, 1)

    ax1.boxplot(
            pred_above,
            showfliers=False,
            patch_artist=True,
            labels=[i for i in neg[1]],
            medianprops=dict(linewidth=2.5, color="black"),
    )
    colors = [color_boxplot[int(np.mean(a) * 20)] for a in pred_above]
    for (i,box) in enumerate(ax1.artists):
        box.set_facecolor(colors[i])
    # ax1.set_title("positives\n(age >= 50 )")
    ax1.set_title(f"Diabetics\n(first diagnostic after {age_separator[1]} years)")
    ax1.set_ylabel("Risk score", fontsize=15)
    ax1.set_xlabel("Age (model input)", fontsize=15)
    ax1.set_ylim(0, 1)

    ax2.boxplot(
            neg[0],
            showfliers=False,
            patch_artist=True,
            labels=[i for i in neg[1]],
            medianprops=dict(linewidth=2.5, color="black"),
    )
    colors = [color_boxplot[int(np.mean(a) * 20)] for a in neg[0]]
    for (i,box) in enumerate(ax2.artists):
        box.set_facecolor(colors[i])
    # ax2.set_title("negatives\n(age >= 60)")
    ax2.set_title("Non-diabetics\n(with 60 years or more)")
    ax2.set_ylabel("Risk score", fontsize=15)
    ax2.set_xlabel("Age (model input)", fontsize=15)
    ax2.set_ylim(0, 1)

    fig.tight_layout(pad=1)

    cmap = mpl.colors.ListedColormap([color_boxplot[i] for i in range(20)])
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbax = fig.add_axes([1.0, 0.13, 0.02, 0.70]) 
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbax, orientation='vertical', ticks=np.arange(0,1.01,.1))
    # cbax.tick_params(labelsize=15)
    cbax.set_ylabel(r'$\mathbb{E}$ [risk score]', fontsize=15)
    
    return fig

def plot_core_family(r):
    db = r.dataset_test_unique
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
            plt.plot(np.transpose(predictions[m_pos_fa, :]), color=COLORS[9])
            plt.scatter(
                (patient_ages[father_mask] - 20) / 5,
                db.predictions[father_mask],
                color=COLORS[9],
            )
        if True in m_pos_mo:
            plt.plot(np.transpose(predictions[m_pos_mo, :]), color=COLORS[6])
            plt.scatter(
                (patient_ages[mother_mask] - 20) / 5,
                db.predictions[mother_mask],
                color=COLORS[6],
            )
        if True in m_pos_chi:
            plt.plot(np.transpose(predictions[m_pos_chi, :]), color=COLORS[7])
            plt.scatter(
                (patient_ages[children_mask] - 20) / 5,
                db.predictions[children_mask],
                color=COLORS[7],
            )
        if True in m_neg_fa:
            plt.plot(np.transpose(predictions[m_neg_fa, :]), color=COLORS[9], ls="--")
            plt.scatter(
                (patient_ages[father_mask] - 20) / 5,
                db.predictions[father_mask],
                color=COLORS[9],
            )
        if True in m_neg_mo:
            plt.plot(np.transpose(predictions[m_neg_mo, :]), color=COLORS[6], ls="--")
            plt.scatter(
                (patient_ages[mother_mask] - 20) / 5,
                db.predictions[mother_mask],
                color=COLORS[6],
            )
        if True in m_neg_chi:
            plt.plot(
                np.transpose(predictions[m_neg_chi, :]), color=COLORS[7], ls="--"
            )
            plt.scatter(
                (patient_ages[children_mask] - 20) / 5,
                db.predictions[children_mask],
                color=COLORS[7],
            )
        # plt.scatter(
        #     (patient_ages[father_mask] - 20) / 5,
        #     db.predictions[father_mask],
        #     c=COLORS[9],
        # )
        # plt.scatter(
        #     (patient_ages[mother_mask] - 20) / 5,
        #     db.predictions[mother_mask],
        #     c=COLORS[6],
        # )
        # plt.scatter(
        #     (patient_ages[children_mask] - 20) / 5,
        #     db.predictions[children_mask],
        #     c=COLORS[7],
        # )
        plt.ylim(0, 1)
        plt.xticks(range(12), np.arange(20, 80, 5))
    plt.tight_layout(pad=1)
    

def plot_family(r):
        db = r.dataset_test_unique
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

        plt.tight_layout(pad=1)
            # break

        # if fig_path != "":
        #     plt.tight_layout(pad=1)
        #     plt.savefig(fig_path)

        # plt.show()

# calibration
def plot_calibration_line(r):
    fig = plt.figure(figsize=(6, 6), dpi=300)

    # Create subplot
    ax = fig.add_subplot(111)

    # Plot calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        r.dataset_test_unique.labels, 
        r.dataset_test_unique.predictions, 
        n_bins=10, 
        strategy='quantile'
    )

    # Plot perfectly calibrated curve
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Plot DiabNet curve
    ax.plot(mean_predicted_value, fraction_of_positives, label="Uncalibrated DiabNet")

    # Get ECE
    ece = r.ece(bootnum=10000, interval='HDI')

    # Get MCE
    mce = r.mce(bootnum=10000, interval='HDI')

    # Get Brier Score
    brier = r.brier(bootnum=10000, interval='HDI')

    # # ECE and MCE legend
    ECE = mpatches.Patch(color=COLORS[1], label='ECE = {:.2f}%'.format(ece['value']*100))
    MCE = mpatches.Patch(color=COLORS[2], label='MCE = {:.2f}%'.format(mce['value']*100))
    BRIER = mpatches.Patch(color=COLORS[3], label='Brier Score = {:.2f}'.format(brier['value']))

    # Configuring plot
    ax.set_title('Calibration curve (Reliability curve)')
    ax.set_ylabel('Fraction of positives')
    ax.set_xlabel('Mean predicted value')

    # Configure legend
    legend1 = plt.legend(
        title='Calibration curves',
        loc="upper left",
    )

    legend2 = plt.legend(
        loc="lower right",
        handles=[ECE, MCE, BRIER]
    )
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    # Display plot
    plt.tight_layout()
    plt.show()

def ece_mce_t_p(preds, targets, bins=10):
    from astropy.stats import bootstrap
    lower_bound = np.arange(0.0, 1.0, 1.0/bins)
    upper_bound = lower_bound + 1.0/bins
    
    ece = np.zeros(1)
    mce = np.zeros(1)
    t = np.zeros(bins)
    p = np.zeros(bins)
    interval_min = np.zeros(bins)
    interval_max = np.zeros(bins)
    
    
    
    for i in range(bins):
        mask = (preds > lower_bound[i]) * (preds <= upper_bound[i])
        if np.any(mask):
#             print(lower_bound[i], upper_bound[i])
#             print(preds[mask])
#             print(targets[mask])
            t[i] = np.mean(targets[mask])
            p[i] = np.mean(preds[mask])
            # num of bootstraps is equal to  
#             data = bootstrap(targets[mask], bootnum=10000, bootfunc=lambda x: np.mean(x) - t[i])
            data = bootstrap(targets[mask], bootnum=10000, bootfunc=lambda x: np.mean(x))
            data = np.sort(data)
#             np.sort(data)
            #print(st.sem(data))
            #interval_min[i], interval_max[i] = st.t.interval(alpha=0.95, df=len(data)-1, loc=t[i], scale=st.sem(data))
#             t[i], _, _, ci = jackknife_stats(targets[mask], np.mean, 0.95)
#             interval_min[i], interval_max[i] = data[249], data[9749]
            # interval_min[i], interval_max[i] = data[249]-t[i], data[9749]-t[i]
            interval_min[i], interval_max[i] = data[499]-t[i], data[9500]-t[i]
            delta = np.abs(np.mean(preds[mask]) - np.mean(targets[mask]))
            ece += delta * np.mean(mask)
            mce = np.maximum(mce, delta)
#             print(ece, mce)
            
#     print(interval_min)
#     print(interval_max)
    return ece, mce, t, p, np.stack([np.abs(interval_min), interval_max])

def plot_calibration(r):
    BINS = 5
    ece, mce, t, p, ece_ci = ece_mce_t_p(r.dataset_test_unique.predictions, r.dataset_test_unique.labels, bins=BINS)
    # fig = plt.figure(figsize=(6, 6), dpi=300)
    fig = plt.figure(figsize=(8,8))


    # Create subplot
    ax = fig.add_subplot(111)

    # Plot calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        r.dataset_test_unique.labels, 
        r.dataset_test_unique.predictions, 
        n_bins=5, 
        strategy='quantile'
    )
    # print(fraction_of_positives)
    # print(mean_predicted_value)

    # Plot perfectly calibrated curve
    ax.plot([0, 1], [0, 1], color=COLORS[7], linewidth=1, linestyle='dashed', label="Perfect calibration")

    # Plot DiabNet curve
    width = 1.0/BINS
    xs = np.arange(width/2,1,width)
    ax.bar(xs, t ,width, yerr=ece_ci, align='center', linewidth=1, alpha=0.99, edgecolor='k', color=COLORS[8], capsize=5.0, ecolor='k', error_kw=dict(mew=3, mfc='g'), label='Observed' )
    #gap
    ax.bar(xs, t-xs, width, xs, align='center', linewidth=0.5, alpha=1.0, edgecolor=COLORS[0], hatch='////', fill=False, label='Gap')
    # ax.bar(mean_predicted_value, fra, width, xs, align='center', linewidth=0.1, alpha=0.5, color='red')
    
    # ax.plot(mean_predicted_value, fraction_of_positives, label="Uncalibrated DiabNet")
    

    # Get ECE
    ece = r.ece(bootnum=10000, interval='HDI')['value']

    # Get MCE
    mce = r.mce(bootnum=10000, interval='HDI')['value']

    # Get Brier Score
    brier = r.brier(bootnum=10000, interval='HDI')['value']
    ax.text(0.61, 0.03, f'ECE = {ece:.3f}\nMCE = {mce:.3f}\nBrier Score = {brier:.3f}', c='white', fontsize=18, horizontalalignment='left', bbox=dict(facecolor='k', ec='k'))

    # # # ECE and MCE legend
    # ECE = mpatches.Patch(color=COLORS[1], label='ECE = {:.2f}%'.format(ece['value']*100))
    # MCE = mpatches.Patch(color=COLORS[2], label='MCE = {:.2f}%'.format(mce['value']*100))
    # BRIER = mpatches.Patch(color=COLORS[3], label='Brier Score = {:.2f}'.format(brier['value']))

    # Configuring plot
    ax.set_title('Reliability diagram')
    ax.set_ylabel('Fraction of positives')
    ax.set_xlabel('Mean predicted value (Confidence)')

    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_yticks(np.arange(0,1.01, 0.1))
    ax.set_xticks(np.arange(0,1.01, 0.1))

    # # Configure legend
    # legend1 = plt.legend(
    #     title='Calibration curves',
    #     loc="upper left",
    # )

    # legend2 = plt.legend(
    #     loc="lower right",
    #     handles=[ECE, MCE, BRIER]
    # )
    # ax.add_artist(legend1)
    # ax.add_artist(legend2)
    ax.legend(fontsize=18)

    # Display plot
    fig.tight_layout()
    # plt.show()
    return fig

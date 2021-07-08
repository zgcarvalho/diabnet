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

COLORS = sns.color_palette("colorblind")

# --------------------------------------------------------------------------- #
# ------------------ Notebook 1: Training results analysis ------------------ #


def get_metrics_from_log(fn: str) -> List[Dict[str, List[float]]]:
    """Get metrics from training log file.
    The metrics are:
        - training loss
        - validation loss
        - balanced accuracy (validation subset)

    Parameters
    ----------
    fn : str
        A path to a log file of DiabNet training procedure.

    Returns
    -------
    ensemble : List[Dict[str, List[float]]]
        A list containing a dictionary of metrics for each model trained in
        the ensemble. The dictionary contains the metrics (training loss,
        validation loss and balanced accuracy) as keys. Each dictionary key
        contains a list of the metric per epoch (Dict[metric] = List[epoch1,
        epoch2, ...]).
    """
    # Create a list for each trained model
    ensemble = []

    # Open the log file
    with open(fn) as f:
        # Read lines into a list
        lines = f.readlines()
        # Iterate
        for line in lines:
            # Add a new item when read a "Model" header in the log file
            if line[0:5] == "Model":
                # The new item is a dictionary with metrics as keys
                ensemble.append(
                    {
                        "training_losses": [],
                        "validation_losses": [],
                        "balanced_accuracy": [],
                    }
                )
            # Read loss in training log
            if line[0] == "T":
                c = line.split()
                ensemble[-1]["training_losses"].append(float(c[4].strip(",")))
            # Read loss and bacc in validation log
            elif line[0] == "V":
                c = line.split()
                ensemble[-1]["validation_losses"].append(float(c[4].strip(",")))
                ensemble[-1]["balanced_accuracy"].append(float(c[8].strip(",")))

    return ensemble


def calculate_stats(
    ensemble: List[Dict[str, List[float]]]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Calculate mean, min5 and max95 for each ensemble metric.

    Parameters
    ----------
    ensemble : List[Dict[str, List[float]]]
        A list containing a dictionary of metrics for each model trained in
        the ensemble. The dictionary contains the metrics (training loss,
        validation loss and balanced accuracy) as keys. Each dictionary key
        contains a list of the metric per epoch (Dict[metric] = List[epoch1,
        epoch2, ...]).

    Returns
    -------
    mean : Dict[str, np.ndarray]
        [description]
    min5 : Dict[str, np.ndarray]
        [description]
    max95 : Dict[str, np.ndarray]
        [description]
    """
    # Create empty dictionaries
    mean = {"training_losses": [], "validation_losses": [], "balanced_accuracy": []}
    min5 = {"training_losses": [], "validation_losses": [], "balanced_accuracy": []}
    max95 = {"training_losses": [], "validation_losses": [], "balanced_accuracy": []}

    # Calculate mean, min5 and max95 for each metric
    for i in range(len(ensemble[0]["training_losses"])):
        for k in ensemble[0].keys():
            mean[k].append(np.mean([e[k][i] for e in ensemble]))
            # is there a better way to use limit this 90% interval?
            min5[k].append(np.percentile([e[k][i] for e in ensemble], 5))
            max95[k].append(np.percentile([e[k][i] for e in ensemble], 95))

    # Convert each key from list to numpy.ndarray
    for k in mean.keys():
        mean[k] = np.asarray(mean[k])
        min5[k] = np.asarray(min5[k])
        max95[k] = np.asarray(max95[k])

    return mean, min5, max95


def plot_loss(data, ax, color) -> None:
    """Plot loss for training and validation subsets.

    Parameters
    ----------
    data
        [description]
    ax
        [description]
    color
        [description]

    Returns
    -------
    ax : [type]
        [description]
    """

    # Get number of epochs
    x = range(len(data[0]["training_losses"]))

    # Prepare data
    y_t, y_tmin, y_tmax = (
        data[0]["training_losses"],
        data[1]["training_losses"],
        data[2]["training_losses"],
    )
    y_v, y_vmin, y_vmax = (
        data[0]["validation_losses"],
        data[1]["validation_losses"],
        data[2]["validation_losses"],
    )

    # Plot values
    ax.plot(y_t, color=color, ls=":", label="train")
    ax.fill_between(x, y_tmin, y_tmax, alpha=0.2, linewidth=0, facecolor=color)
    ax.plot(y_v, color=color, label="validation")
    ax.fill_between(x, y_vmin, y_vmax, alpha=0.2, linewidth=0, facecolor=color)

    # Labels
    plt.ylabel("Loss", fontsize=9)

    # Limits
    plt.xlim(-3, 303)

    # Ticks
    plt.yticks(np.arange(0.4, 1.05, 0.2), fontsize=9)
    plt.xticks(fontsize=9)
    plt.tick_params(length=1)

    return ax


def plot_bacc(data, ax, colors):
    """Plot balanced accuracy.

    Parameters
    ----------
    datas : [type]
        [description]
    ax : [type]
        [description]
    colors : [type]
        [description]

    Returns
    -------
    ax : [type]
        [description]
    """

    for i in range(len(data)):
        # Get number of epochs
        x = range(len(data[i][0]["balanced_accuracy"]))

        # Prepare data
        y, y_min, y_max = (
            data[i][0]["balanced_accuracy"] * 100,
            data[i][1]["balanced_accuracy"] * 100,
            data[i][2]["balanced_accuracy"] * 100,
        )
        # Plot data
        ax.plot(y, color=colors[i])
        ax.fill_between(x, y_min, y_max, alpha=0.2, linewidth=0.0, facecolor=colors[i])

    # Labels
    plt.xlabel("epoch", fontsize=9)
    plt.ylabel("Balanced accuracy (%)", fontsize=9)

    # Limits
    plt.ylim(40, 100)
    plt.xlim(-3, 303)

    # Ticks
    plt.yticks(np.arange(40, 101, 10), fontsize=9)
    plt.xticks(fontsize=9)

    return ax


# --------------------------------------------------------------------------- #
# ------------------- Notebook 2: Report metrics ensemble ------------------- #

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
            "AUC": [r.auc, r.auc_neg_older50],
            "AvgPrec": [r.average_precision, r.average_precision_neg_older50],
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
    fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, dpi=300)
    ax0.set_title("Full test set", fontsize=14)
    ax1.set_title("Subset without\nnegatives younger than 50", fontsize=10)
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


def plot_roc(r, interval="HDI"):
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax1 = fig.add_subplot(111)
    c = [COLORS[2], "gainsboro"]
    _roc_b(
        r.dataset_test_unique, ax1, num=2000, alpha=0.05, interval=interval, colors=c
    )
    # return fig


def plot_roc_neg_older50(r, interval="HDI"):
    fig = plt.figure(figsize=(6, 6), dpi=300)
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
    # return fig


def plot_roc_comp(r, interval="HDI"):
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax1 = fig.add_subplot(111)
    _roc_b_comp(
        r.dataset_test_unique,
        r.dataset_test_unique_subset_neg_older50,
        ax1,
        num=1000,
        alpha=0.05,
        interval=interval,
    )
    # return fig


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
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")


def plot_precision_recall(r, interval="HDI"):
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax1 = fig.add_subplot(111)
    c = [COLORS[2], "gainsboro"]
    _precision_recall_b(
        r.dataset_test_unique, ax1, num=2000, alpha=0.05, interval=interval, colors=c
    )
    # return fig


def plot_precision_recall_neg_older50(r, interval="HDI"):
    fig = plt.figure(figsize=(6, 6), dpi=300)
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


def plot_precision_recall_comp(r, interval="HDI"):
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax1 = fig.add_subplot(111)
    _precision_recall_b_comp(
        r.dataset_test_unique,
        r.dataset_test_unique_subset_neg_older50,
        ax1,
        num=1000,
        alpha=0.05,
        interval=interval,
    )
    # return fig


# --------------------------------------------------------------------------- #
# -------------- Notebook 4: Family Cross Validation analysis --------------- #


def _data2barplot(
    data: pd.DataFrame, selection: Optional[List[str]] = None, orient: str = "index"
) -> Tuple[Dict[str, List[float]], List[str]]:
    """Prepara pandas.DataFrame to barplot.

    Parameters
    ----------
    data : pd.DataFrame
        Data of family cross-validation analysis.
    selection : List[str], optional
        A list of metrics to be ploted. I must only contain
        header in data, by default None.
    orient : str {'dict', 'index'}, optional
        Determines the type of the values of the dictionary, by default `index`.

            * 'dict' : dict like {column -> {index -> value}}

            * 'index' (default) : dict like {index -> {column -> value}}

    Returns
    -------
    data : Dict[str, List[float]]
        Data prepared to barplot function.
    labels : List[str]
        A list of labels to barplot function.

    Raises
    ------
    TypeError
        "`data` must be a pandas.DataFrame.
    ValueError
        `selection` must be columns of `data`.
    TypeError
        `orient` must be `dict` or `index`.
    ValueError
        `orient` must be `dict` or `index`.
    """
    # Check arguments
    if type(data) not in [pd.DataFrame]:
        raise TypeError("`data` must be a pandas.DataFrame.")
    if selection is not None:
        if type(selection) not in [list]:
            raise TypeError("`selection` must be a list.")
        elif not set(selection).issubset(list(data.columns)):
            raise ValueError("`selection` must be columns of `data`.")
        # Apply selection
        data = data[selection]
    if type(orient) not in [str]:
        raise TypeError("`orient` must be `dict` or `index`.")
    elif orient not in ["dict", "index"]:
        raise ValueError("`orient` must be `dict` or `index`.")

    # Convert data to dict
    tmp = data.to_dict(orient)

    # Get labels
    labels = list(tmp[list(tmp)[0]].keys())

    # Prepare data for barplot
    data = {key: list(tmp[key].values()) for key in tmp.keys()}

    return data, labels


def barplot(
    ax,
    data: Dict[str, List[float]],
    selection: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    rotation: int = 0,
    ha: str = "center",
    orient: str = "index",
    colors: Optional[Union[List[float], seaborn.palettes._ColorPalette]] = None,
    total_width: float = 0.9,
    single_width: float = 1,
    legend: bool = True,
):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.
    data : Dict[str, List[float]]
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.
    selection : List[str], optional
        A list of metrics to be ploted. I must only contain
        header in data, by default None.
    labels : List[str], optional
        A list of labels to use as axis ticks. If we want to remove xticks, set
        to []. If None, use labels from _data2barplot, by default None.
    rotation : int, optional
        Rotation (degrees) of xticks, by default 0.
    ha : str {'left', 'center', 'right'}, optional
        Horizontal aligments of xticks, by default `center`.
    orient : str {'dict', 'index'}, optional
        Determines the type of the values of the dictionary, by default `index`.

            * 'dict' : dict like {column -> {index -> value}}

            * 'index' (default) : dict like {index -> {column -> value}}
    colors : Union[List[float], seaborn.palettes._ColorPalette], optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cylel, by default None.
    total_width : float, optional
        The width of a bar group. 0.9 means that 90% of the x-axis is covered
        by bars and 10% will be spaces between the bars, by default 0.9.
    single_width: float, optional
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner, by default 1.
    legend: bool, optional
        If this is set to true, a legend will be added to the axis, by default True.
    """
    from matplotlib import pyplot as plt

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Prepare data
    if labels is None:
        data, labels = _data2barplot(data, selection=selection, orient=orient)
    else:
        data, _ = _data2barplot(data, selection=selection, orient=orient)

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(
                x + x_offset,
                y,
                width=bar_width * single_width,
                color=colors[i % len(colors)],
            )

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Set x ticks
    ax.set_xticks(range(len(labels)), minor=False)
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())


# --------------------------------------------------------------------------- #
# ----------------- Notebook 7: Feature importance analysis ----------------- #


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
                imp[self.feat_names[i]].append(
                    attr[:, 1, i][mask[:, i]].detach().numpy()
                )

        for i in range(1000):
            imp[self.feat_names[i]] = np.concatenate(imp[self.feat_names[i]])

        return imp

    def attr_snps_mean(self, attrs, mask):
        imp = {}
        for i in range(1000):
            imp[self.feat_names[i]] = np.concatenate(
                [attr[:, 1, i][mask[:, i]].detach().numpy() for attr in attrs]
            )

        return imp

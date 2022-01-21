# import torch
# import seaborn
# import seaborn as sns
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Union, Tuple, Optional
# from collections import OrderedDict
# import matplotlib.pyplot as plt
# from matplotlib.ticker import AutoMinorLocator
# from sklearn.utils.validation import indexable
# from captum.attr import IntegratedGradients
# from diabnet.data import encode_features



# COLORS = sns.color_palette("colorblind")

# --------------------------------------------------------------------------- #
# ------------------ Notebook 1: Training results analysis ------------------ #


# def get_metrics_from_log(fn: str) -> List[Dict[str, List[float]]]:
#     """Get metrics from training log file.
#     The metrics are:
#         - training loss
#         - validation loss
#         - balanced accuracy (validation subset)

#     Parameters
#     ----------
#     fn : str
#         A path to a log file of DiabNet training procedure.

#     Returns
#     -------
#     ensemble : List[Dict[str, List[float]]]
#         A list containing a dictionary of metrics for each model trained in
#         the ensemble. The dictionary contains the metrics (training loss,
#         validation loss and balanced accuracy) as keys. Each dictionary key
#         contains a list of the metric per epoch (Dict[metric] = List[epoch1,
#         epoch2, ...]).
#     """
#     # Create a list for each trained model
#     ensemble = []

#     # Open the log file
#     with open(fn) as f:
#         # Read lines into a list
#         lines = f.readlines()
#         # Iterate
#         for line in lines:
#             # Add a new item when read a "Model" header in the log file
#             if line[0:5] == "Model":
#                 # The new item is a dictionary with metrics as keys
#                 ensemble.append(
#                     {
#                         "training_losses": [],
#                         "validation_losses": [],
#                         "balanced_accuracy": [],
#                     }
#                 )
#             # Read loss in training log
#             if line[0] == "T":
#                 c = line.split()
#                 ensemble[-1]["training_losses"].append(float(c[4].strip(",")))
#             # Read loss and bacc in validation log
#             elif line[0] == "V":
#                 c = line.split()
#                 ensemble[-1]["validation_losses"].append(float(c[4].strip(",")))
#                 ensemble[-1]["balanced_accuracy"].append(float(c[8].strip(",")))

#     return ensemble

# def calculate_stats(
#     ensemble: List[Dict[str, List[float]]]
# ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
#     """Calculate mean, min5 and max95 for each ensemble metric.

#     Parameters
#     ----------
#     ensemble : List[Dict[str, List[float]]]
#         A list containing a dictionary of metrics for each model trained in
#         the ensemble. The dictionary contains the metrics (training loss,
#         validation loss and balanced accuracy) as keys. Each dictionary key
#         contains a list of the metric per epoch (Dict[metric] = List[epoch1,
#         epoch2, ...]).

#     Returns
#     -------
#     mean : Dict[str, np.ndarray]
#         [description]
#     min5 : Dict[str, np.ndarray]
#         [description]
#     max95 : Dict[str, np.ndarray]
#         [description]
#     """
#     # Create empty dictionaries
#     mean = {"training_losses": [], "validation_losses": [], "balanced_accuracy": []}
#     min5 = {"training_losses": [], "validation_losses": [], "balanced_accuracy": []}
#     max95 = {"training_losses": [], "validation_losses": [], "balanced_accuracy": []}

#     # Calculate mean, min5 and max95 for each metric
#     for i in range(len(ensemble[0]["training_losses"])):
#         for k in ensemble[0].keys():
#             mean[k].append(np.mean([e[k][i] for e in ensemble]))
#             # is there a better way to use limit this 90% interval?
#             min5[k].append(np.percentile([e[k][i] for e in ensemble], 5))
#             max95[k].append(np.percentile([e[k][i] for e in ensemble], 95))

#     # Convert each key from list to numpy.ndarray
#     for k in mean.keys():
#         mean[k] = np.asarray(mean[k])
#         min5[k] = np.asarray(min5[k])
#         max95[k] = np.asarray(max95[k])

#     return mean, min5, max95

# def plot_loss(data, ax, color) -> None:
#     """Plot loss for training and validation subsets.

#     Parameters
#     ----------
#     data
#         [description]
#     ax
#         [description]
#     color
#         [description]

#     Returns
#     -------
#     ax : [type]
#         [description]
#     """

#     # Get number of epochs
#     x = range(len(data[0]["training_losses"]))

#     # Prepare data
#     y_t, y_tmin, y_tmax = (
#         data[0]["training_losses"],
#         data[1]["training_losses"],
#         data[2]["training_losses"],
#     )
#     y_v, y_vmin, y_vmax = (
#         data[0]["validation_losses"],
#         data[1]["validation_losses"],
#         data[2]["validation_losses"],
#     )

#     # Plot values
#     ax.plot(y_t, color=color, ls=":", label="train")
#     ax.fill_between(x, y_tmin, y_tmax, alpha=0.2, linewidth=0, facecolor=color)
#     ax.plot(y_v, color=color, label="validation")
#     ax.fill_between(x, y_vmin, y_vmax, alpha=0.2, linewidth=0, facecolor=color)

#     # Labels

#     plt.ylabel("Loss")

#     # Limits
#     plt.xlim(-3, 303)

#     # Ticks
#     plt.yticks(np.arange(0.62, 0.75, 0.1))
#     plt.tick_params(length=1)

#     return ax

# def plot_bacc(data, ax, colors):
#     """Plot balanced accuracy.

#     Parameters
#     ----------
#     datas : [type]
#         [description]
#     ax : [type]
#         [description]
#     colors : [type]
#         [description]

#     Returns
#     -------
#     ax : [type]
#         [description]
#     """

#     for i in range(len(data)):
#         # Get number of epochs
#         x = range(len(data[i][0]["balanced_accuracy"]))

#         # Prepare data
#         y, y_min, y_max = (
#             data[i][0]["balanced_accuracy"] * 100,
#             data[i][1]["balanced_accuracy"] * 100,
#             data[i][2]["balanced_accuracy"] * 100,
#         )
#         # Plot data
#         ax.plot(y, color=colors[i])
#         ax.fill_between(x, y_min, y_max, alpha=0.2, linewidth=0.0, facecolor=colors[i])

#     # Labels
#     plt.xlabel("epoch")
#     plt.ylabel("Balanced accuracy (%)")

#     # Limits
#     plt.ylim(40, 100)
#     plt.xlim(-3, 303)

#     # Ticks
#     plt.yticks(np.arange(40, 101, 10))


#     return ax

# --------------------------------------------------------------------------- #
# ------------------- Notebook 2: Report metrics ensemble ------------------- #

# import matplotlib as mpl
# from sklearn.metrics import (
#     roc_auc_score,
#     roc_curve,
#     precision_recall_curve,
#     # f1_score,
#     # accuracy_score,
#     # balanced_accuracy_score,
#     # brier_score_loss,
#     average_precision_score,
#     # precision_score,
#     # recall_score,  # if pos_label = (sensitivity or TPR) if neg_label = (specificity or TNR)
# )
# from sklearn.neighbors import KernelDensity
# from scipy.stats import norm
# from sklearn.calibration import calibration_curve
# import matplotlib.patches as mpatches

# def plot_metrics(r, interval="HDI"):
#     """Plot testing metrics.
#     Parameters
#     ----------
#     r: DiabNetReport

#     Returns
#     -------
#     figure
#     """
#     metrics = OrderedDict(
#         {
#             "AUROC": [r.auc, r.auc_neg_older50],
#             "AUPRC": [r.average_precision, r.average_precision_neg_older50],
#             "BACC": [r.bacc, r.bacc_neg_older50],
#             # 'ACC': [r.acc, r.acc_neg_older50],
#             "F1": [r.f1, r.f1_neg_older50],
#             "Precision": [r.precision, r.precision_neg_older50],
#             "Sensitivity": [r.sensitivity, r.sensitivity_neg_older50],
#             "Specificity": [r.specificity, r.specificity_neg_older50],
#             "Brier": [r.brier, r.brier_neg_older50],
#             "ECE": [r.ece, r.ece_neg_older50],
#             "MCE": [r.mce, r.mce_neg_older50],
#         }
#     )
#     co = sns.color_palette("coolwarm", n_colors=33)
#     # fig = plt.figure(dpi=300)
#     # ax0 = fig.add_subplot(221)
#     # ax1 = fig.add_subplot(222)
#     fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})
#     ax0.set_title("Full test set", fontsize=15)
#     ax1.set_title("Subset without\nnegatives younger than 50", fontsize=14)
#     for (i, k) in enumerate(metrics.keys()):
#         v = metrics[k][0](bootnum=1000, interval=interval)
#         ax0.errorbar(
#             v["value"],
#             i,
#             xerr=[[v["value"] - v["lower"]], [v["upper"] - v["value"]]],
#             fmt=".",
#             capsize=3.0,
#             color=co[int(v["value"] * 33)],
#         )
#         vo = metrics[k][1](bootnum=1000, interval=interval)
#         ax1.errorbar(
#             vo["value"],
#             i,
#             xerr=[[vo["value"] - vo["lower"]], [vo["upper"] - vo["value"]]],
#             fmt=".",
#             capsize=3.0,
#             color=co[int(vo["value"] * 33)],
#         )

#     ax0.set_xticks(np.arange(0, 1.01, 0.2))
#     ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
#     ax1.xaxis.grid(True, which="minor")
#     ax1.set_xticks(np.arange(0, 1.01, 0.2))
#     ax0.xaxis.set_minor_locator(AutoMinorLocator(2))
#     ax0.xaxis.grid(True, which="minor")
#     ax0.set_ylim(len(metrics), -1)
#     ax0.set_yticks(range(len(metrics)))
#     ax0.set_yticklabels(metrics.keys())
#     ax1.set_ylim(len(metrics), -1)
#     ax1.set_yticks(range(len(metrics)))
#     ax1.set_yticklabels("")

#     conf_matrix = r.confusion()
#     conf_matrix_50 = r.confusion_neg_older50()
#     ax2.matshow(conf_matrix, cmap=plt.cm.Purples, alpha=0.3)
#     for i in range(conf_matrix.shape[0]):
#         for j in range(conf_matrix.shape[1]):
#             ax2.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large')

#     ax3.matshow(conf_matrix_50, cmap=plt.cm.Purples, alpha=0.3)
#     for i in range(conf_matrix_50.shape[0]):
#         for j in range(conf_matrix_50.shape[1]):
#             ax3.text(x=j, y=i,s=conf_matrix_50[i, j], va='center', ha='center', size='large')

#     ax2.set_xlabel("Predicted", fontsize=18)
#     ax2.set_ylabel("Actual", fontsize=18)
#     ax2.set_yticks(range(2))
#     ax2.set_yticklabels(["N", "P"])
#     ax2.set_xticks(range(2))
#     ax2.set_xticklabels(["N", "P"])
#     ax2.xaxis.tick_bottom()
#     ax2.grid(False)

#     ax3.set_xlabel("Predicted", fontsize=18)
#     ax3.set_ylabel("Actual", fontsize=18)
#     ax3.set_yticks(range(2))
#     ax3.set_yticklabels(["N", "P"])
#     ax3.set_xticks(range(2))
#     ax3.set_xticklabels(["N", "P"])
#     ax3.xaxis.tick_bottom()
#     ax3.grid(False)

#     # print()
#     # print(r.confusion_neg_older50())

#     return fig

# def _hdi_idx(data, p):
#     """Highest Density Interval"""
#     l = int(len(data) * p)
#     diff, lower, upper = data[-1] - data[0], data[0], data[-1]
#     for i in range(len(data) - l):
#         new_diff = data[i + l - 1] - data[i]
#         if diff > new_diff:
#             diff = new_diff
#             lower = data[i]
#             upper = data[i + l - 1]
#             lower_idx = i
#             upper_idx = i + l - 1
#     return (lower_idx, upper_idx)

# def _roc_b(
#     db, ax, num=1000, alpha=0.05, interval="CI", colors=[COLORS[2], "gainsboro"]
# ):

#     # bootstrap with ensemble
#     n_patients = len(db.labels)
#     n_models = len(db.predictions_ensemble[0])

#     aucs = np.empty(num)
#     data = np.empty((num, 2, n_patients))

#     for i in range(num):
#         mask_preds = np.random.choice(n_patients, n_patients)
#         mask_ensemble = np.random.choice(n_models, n_models)

#         aucs[i] = roc_auc_score(
#             db.labels[mask_preds],
#             np.mean(db.predictions_ensemble[mask_preds][:, mask_ensemble], 1),
#         )
#         data[i, 0] = db.labels[mask_preds]
#         data[i, 1] = np.mean(db.predictions_ensemble[mask_preds][:, mask_ensemble], 1)

#     sorting_idx = np.argsort(aucs)

#     if interval == "HDI":
#         lower_idx, upper_idx = _hdi_idx(aucs[sorting_idx], 1 - alpha)
#     else:
#         # CDI
#         lower_idx = int(num * alpha / 2)
#         upper_idx = int(num * (1 - alpha / 2))

#     idx = sorting_idx[lower_idx:upper_idx]

#     for i in idx:
#         fpr, tpr, _ = roc_curve(data[i, 0], data[i, 1])
#         ax.plot(fpr, tpr, color=colors[1], alpha=0.5, linewidth=0.2)

#     fpr, tpr, _ = roc_curve(db.labels, db.predictions)
#     ax.plot(fpr, tpr, color=colors[0])
#     ax.text(0.7, 0.08, f'AUROC: {np.mean(aucs):.3f}', fontsize=21, horizontalalignment='center')
#     ax.text(0.7, 0.05, f'{int((1-alpha)*100)}% {interval}: [{aucs[idx[0]]:.3f}, {aucs[idx[-1]]:.3f}]', fontsize=13, horizontalalignment='center')


#     ax.plot([0, 1], [0, 1], color=COLORS[7], linewidth=0.5)
#     ax.set_ylim(0, 1)
#     ax.set_xlim(0, 1)
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")

# def _roc_b_comp(
#     db1,
#     db2,
#     ax,
#     num=1000,
#     alpha=0.05,
#     interval="CI",
#     colors=[[COLORS[2], "palegreen"], [COLORS[0], "lightskyblue"]],
# ):

#     # bootstrap with ensemble
#     n_patients_1 = len(db1.labels)
#     n_patients_2 = len(db2.labels)
#     # n_models are the same for db1 and db2
#     n_models = len(db1.predictions_ensemble[0])

#     aucs_1 = np.empty(num)
#     aucs_2 = np.empty(num)
#     data_1 = np.empty((num, 2, n_patients_1))
#     data_2 = np.empty((num, 2, n_patients_2))

#     for i in range(num):
#         mask_preds_1 = np.random.choice(n_patients_1, n_patients_1)
#         mask_preds_2 = np.random.choice(n_patients_2, n_patients_2)
#         mask_ensemble_1 = np.random.choice(n_models, n_models)
#         mask_ensemble_2 = np.random.choice(n_models, n_models)

#         aucs_1[i] = roc_auc_score(
#             db1.labels[mask_preds_1],
#             np.mean(db1.predictions_ensemble[mask_preds_1][:, mask_ensemble_1], 1),
#         )
#         aucs_2[i] = roc_auc_score(
#             db2.labels[mask_preds_2],
#             np.mean(db2.predictions_ensemble[mask_preds_2][:, mask_ensemble_2], 1),
#         )
#         data_1[i, 0] = db1.labels[mask_preds_1]
#         data_2[i, 0] = db2.labels[mask_preds_2]
#         data_1[i, 1] = np.mean(
#             db1.predictions_ensemble[mask_preds_1][:, mask_ensemble_1], 1
#         )
#         data_2[i, 1] = np.mean(
#             db2.predictions_ensemble[mask_preds_2][:, mask_ensemble_2], 1
#         )

#     sorting_idx_1 = np.argsort(aucs_1)
#     sorting_idx_2 = np.argsort(aucs_2)

#     if interval == "HDI":
#         lower_idx_1, upper_idx_1 = _hdi_idx(aucs_1[sorting_idx_1], 1 - alpha)
#         lower_idx_2, upper_idx_2 = _hdi_idx(aucs_2[sorting_idx_2], 1 - alpha)
#     else:
#         # CDI
#         lower_idx_1 = int(num * alpha / 2)
#         lower_idx_2 = lower_idx_1
#         upper_idx_1 = int(num * (1 - alpha / 2))
#         upper_idx_2 = upper_idx_1

#     idx_1 = sorting_idx_1[lower_idx_1:upper_idx_1]
#     idx_2 = sorting_idx_2[lower_idx_2:upper_idx_2]

#     for i in range(len(idx_1)):
#         fpr, tpr, _ = roc_curve(data_1[idx_1[i], 0], data_1[idx_1[i], 1])
#         ax.plot(fpr, tpr, color=colors[0][1], alpha=0.3, linewidth=0.1)
#         fpr, tpr, _ = roc_curve(data_2[idx_2[i], 0], data_2[idx_2[i], 1])
#         ax.plot(fpr, tpr, color=colors[1][1], alpha=0.2, linewidth=0.1)

#     fpr, tpr, _ = roc_curve(db1.labels, db1.predictions)
#     ax.plot(fpr, tpr, color=colors[0][0])
#     fpr, tpr, _ = roc_curve(db2.labels, db2.predictions)
#     ax.plot(fpr, tpr, color=colors[1][0])

#     ax.plot([0, 1], [0, 1], color=COLORS[7], linewidth=0.5)
#     ax.set_ylim(0, 1)
#     ax.set_xlim(0, 1)
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")

# def _roc_by_ages(db, ax):
#     colors = sns.color_palette("BrBG_r", n_colors=12)
#     for age, preds in db.predictions_per_age.items():
#         fpr, tpr, _ = roc_curve(db.labels, preds)
#         score = roc_auc_score(db.labels, preds)
#         ax.plot(fpr, tpr, color=colors[int((age-20)/5)],label=f"{age} : {score:.3f}")
        

#     ax.legend(
#         title='Age : AUROC',
#         title_fontsize='large',
#         loc="lower right",
#         prop={'size': 13},
#     )
#     ax.plot([0, 1], [0, 1], color=COLORS[7], linewidth=0.5)
#     ax.set_ylim(0, 1)
#     ax.set_xlim(0, 1)
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")

# def plot_roc_by_age(r):
#     # fig = plt.figure(figsize=(6, 6), dpi=300)
#     fig = plt.figure(figsize=(8, 8))
#     ax1 = fig.add_subplot(111)
#     _roc_by_ages(r.dataset_test_unique, ax1)
#     return fig

# def plot_roc_by_age_neg_older50(r):
#     # fig = plt.figure(figsize=(6, 6), dpi=300)
#     fig = plt.figure(figsize=(8, 8))
#     ax1 = fig.add_subplot(111)
#     _roc_by_ages(r.dataset_test_unique_subset_neg_older50, ax1)
#     return fig

# def plot_roc(r, interval="HDI"):
#     # fig = plt.figure(figsize=(6, 6), dpi=300)
#     fig = plt.figure(figsize=(8, 8))
#     ax1 = fig.add_subplot(111)
#     c = [COLORS[2], "gainsboro"]
#     _roc_b(
#         r.dataset_test_unique, ax1, num=2000, alpha=0.05, interval=interval, colors=c
#     )
#     return fig

# def plot_roc_neg_older50(r, interval="HDI"):
#     # fig = plt.figure(figsize=(6, 6), dpi=300)
#     fig = plt.figure(figsize=(8, 8))
#     ax1 = fig.add_subplot(111)
#     c = [COLORS[0], "gainsboro"]
#     _roc_b(
#         r.dataset_test_unique_subset_neg_older50,
#         ax1,
#         num=2000,
#         alpha=0.05,
#         interval=interval,
#         colors=c,
#     )
#     return fig

# def plot_roc_comp(r, interval="HDI"):
#     # fig = plt.figure(figsize=(6, 6), dpi=300)
#     fig = plt.figure(figsize=(8, 8))
#     ax1 = fig.add_subplot(111)
#     _roc_b_comp(
#         r.dataset_test_unique,
#         r.dataset_test_unique_subset_neg_older50,
#         ax1,
#         num=1000,
#         alpha=0.05,
#         interval=interval,
#     )
#     return fig

# def _precision_recall_b(
#     db, ax, num=1000, alpha=0.05, interval="CI", colors=[COLORS[2], "gainsboro"]
# ):

#     # bootstrap with ensemble
#     n_patients = len(db.labels)
#     n_models = len(db.predictions_ensemble[0])

#     avgps = np.empty(num)
#     data = np.empty((num, 2, n_patients))

#     for i in range(num):
#         mask_preds = np.random.choice(n_patients, n_patients)
#         mask_ensemble = np.random.choice(n_models, n_models)

#         avgps[i] = average_precision_score(
#             db.labels[mask_preds],
#             np.mean(db.predictions_ensemble[mask_preds][:, mask_ensemble], 1),
#         )
#         data[i, 0] = db.labels[mask_preds]
#         data[i, 1] = np.mean(db.predictions_ensemble[mask_preds][:, mask_ensemble], 1)

#     sorting_idx = np.argsort(avgps)

#     if interval == "HDI":
#         lower_idx, upper_idx = _hdi_idx(avgps[sorting_idx], 1 - alpha)
#     else:
#         # CDI
#         lower_idx = int(num * alpha / 2)
#         upper_idx = int(num * (1 - alpha / 2))

#     idx = sorting_idx[lower_idx:upper_idx]

#     for i in idx:
#         precision, recall, _ = precision_recall_curve(data[i, 0], data[i, 1])
#         ax.step(
#             recall,
#             precision,
#             where="post",
#             color=colors[1],
#             alpha=0.5,
#             linewidth=0.2,
#         )

#     precision, recall, _ = precision_recall_curve(db.labels, db.predictions)
#     ax.step(recall, precision, where="post", color=colors[0])
#     ax.text(0.3, 0.08, f'AUPRC: {np.mean(avgps):.3f}', fontsize=21, horizontalalignment='center')
#     ax.text(0.3, 0.05, f'{int((1-alpha)*100)}% {interval}: [{avgps[idx[0]]:.3f}, {avgps[idx[-1]]:.3f}]', fontsize=13, horizontalalignment='center')

#     ax.plot([0, 1], [0, 1], color=COLORS[7], linewidth=0.5)
#     ax.set_ylim(0, 1)
#     ax.set_xlim(0, 1)
#     ax.set_xlabel("Recall")
#     ax.set_ylabel("Precision")

# def _precision_recall_b_comp(
#     db1,
#     db2,
#     ax,
#     num=1000,
#     alpha=0.05,
#     interval="CI",
#     colors=[[COLORS[2], "palegreen"], [COLORS[0], "lightskyblue"]],
# ):

#     # bootstrap with ensemble
#     n_patients_1 = len(db1.labels)
#     n_patients_2 = len(db2.labels)
#     # n_models are the same for db1 and db2
#     n_models = len(db1.predictions_ensemble[0])

#     avgs_1 = np.empty(num)
#     avgs_2 = np.empty(num)
#     data_1 = np.empty((num, 2, n_patients_1))
#     data_2 = np.empty((num, 2, n_patients_2))

#     for i in range(num):
#         mask_preds_1 = np.random.choice(n_patients_1, n_patients_1)
#         mask_preds_2 = np.random.choice(n_patients_2, n_patients_2)
#         mask_ensemble_1 = np.random.choice(n_models, n_models)
#         mask_ensemble_2 = np.random.choice(n_models, n_models)

#         avgs_1[i] = average_precision_score(
#             db1.labels[mask_preds_1],
#             np.mean(db1.predictions_ensemble[mask_preds_1][:, mask_ensemble_1], 1),
#         )
#         avgs_2[i] = average_precision_score(
#             db2.labels[mask_preds_2],
#             np.mean(db2.predictions_ensemble[mask_preds_2][:, mask_ensemble_2], 1),
#         )
#         data_1[i, 0] = db1.labels[mask_preds_1]
#         data_2[i, 0] = db2.labels[mask_preds_2]
#         data_1[i, 1] = np.mean(
#             db1.predictions_ensemble[mask_preds_1][:, mask_ensemble_1], 1
#         )
#         data_2[i, 1] = np.mean(
#             db2.predictions_ensemble[mask_preds_2][:, mask_ensemble_2], 1
#         )

#     sorting_idx_1 = np.argsort(avgs_1)
#     sorting_idx_2 = np.argsort(avgs_2)

#     if interval == "HDI":
#         lower_idx_1, upper_idx_1 = _hdi_idx(avgs_1[sorting_idx_1], 1 - alpha)
#         lower_idx_2, upper_idx_2 = _hdi_idx(avgs_2[sorting_idx_2], 1 - alpha)
#     else:
#         # CDI
#         lower_idx_1 = int(num * alpha / 2)
#         lower_idx_2 = lower_idx_1
#         upper_idx_1 = int(num * (1 - alpha / 2))
#         upper_idx_2 = upper_idx_1

#     idx_1 = sorting_idx_1[lower_idx_1:upper_idx_1]
#     idx_2 = sorting_idx_2[lower_idx_2:upper_idx_2]

#     for i in range(len(idx_1)):
#         precision, recall, _ = precision_recall_curve(
#             data_1[idx_1[i], 0], data_1[idx_1[i], 1]
#         )
#         ax.step(
#             recall,
#             precision,
#             where="post",
#             color=colors[0][1],
#             alpha=0.2,
#             linewidth=0.1,
#         )
#         precision, recall, _ = precision_recall_curve(
#             data_2[idx_2[i], 0], data_2[idx_2[i], 1]
#         )
#         ax.step(
#             recall,
#             precision,
#             where="post",
#             color=colors[1][1],
#             alpha=0.3,
#             linewidth=0.1,
#         )

#     precision, recall, _ = precision_recall_curve(db1.labels, db1.predictions)
#     ax.step(recall, precision, where="post", color=colors[0][0])
#     precision, recall, _ = precision_recall_curve(db2.labels, db2.predictions)
#     ax.step(recall, precision, where="post", color=colors[1][0])

#     ax.plot([0, 1], [0, 1], color=COLORS[7], linewidth=0.5)
#     ax.set_ylim(0, 1)
#     ax.set_xlim(0, 1)
#     ax.set_xlabel("Recall")
#     ax.set_ylabel("Precision")

# def _precision_recall_by_ages(db, ax):
#     colors = sns.color_palette("BrBG_r", n_colors=12)
#     for age, preds in db.predictions_per_age.items():
#         precision, recall, _ = precision_recall_curve(db.labels, preds)
#         score = average_precision_score(db.labels, preds)
#         ax.step(
#             recall,
#             precision,
#             where="post",
#             color=colors[int((age-20)/5)],
#             # alpha=0.5,
#             # linewidth=0.2,
#             label=f"{age} : {score:.3f}"
#         )

#     ax.legend(
#         title='Age : AUPRC',
#         title_fontsize='large',
#         loc="lower left",
#         prop={'size': 13},
#     )
#     ax.set_ylim(0, 1)
#     ax.set_xlim(0, 1)
#     ax.set_xlabel("Recall")
#     ax.set_ylabel("Precision")

# def plot_precision_recall_by_age(r):
#     # fig = plt.figure(figsize=(6, 6), dpi=300)
#     fig = plt.figure(figsize=(8, 8))
#     ax1 = fig.add_subplot(111)
#     _precision_recall_by_ages(r.dataset_test_unique, ax1)
#     return fig

# def plot_precision_recall_by_age_neg_older50(r):
#     # fig = plt.figure(figsize=(6, 6), dpi=300)
#     fig = plt.figure(figsize=(8, 8))
#     ax1 = fig.add_subplot(111)
#     _precision_recall_by_ages(r.dataset_test_unique_subset_neg_older50, ax1)
#     return fig

# def plot_precision_recall(r, interval="HDI"):
#     # fig = plt.figure(figsize=(6, 6), dpi=300)
#     fig = plt.figure(figsize=(8, 8))
#     ax1 = fig.add_subplot(111)
#     c = [COLORS[2], "gainsboro"]
#     _precision_recall_b(
#         r.dataset_test_unique, ax1, num=2000, alpha=0.05, interval=interval, colors=c
#     )
#     return fig

# def plot_precision_recall_neg_older50(r, interval="HDI"):
#     # fig = plt.figure(figsize=(6, 6), dpi=300)
#     fig = plt.figure(figsize=(8, 8))
#     ax1 = fig.add_subplot(111)
#     c = [COLORS[0], "gainsboro"]
#     _precision_recall_b(
#         r.dataset_test_unique_subset_neg_older50,
#         ax1,
#         num=2000,
#         alpha=0.05,
#         interval=interval,
#         colors=c,
#     )
#     return fig

# def plot_precision_recall_comp(r, interval="HDI"):
#     # fig = plt.figure(figsize=(6, 6), dpi=300)
#     fig = plt.figure(figsize=(8, 8))
#     ax1 = fig.add_subplot(111)
#     _precision_recall_b_comp(r.dataset_test_unique, r.dataset_test_unique_subset_neg_older50, ax1, num=1000, alpha=0.05, interval=interval)
#     return fig

# def _kde_plot(x,ax,color):
#     if len(x) > 100:
#         x = x[::100]
#     x_d = np.linspace(0, 1, 1000)

#     kde = KernelDensity(bandwidth=0.17, kernel='gaussian', metric='euclidean')
#     kde.fit(x[:, None])

#     # score_samples returns the log of the probability density
#     logprob = kde.score_samples(x_d[:, None])
#     prob = np.exp(logprob)

#     ax.plot(x_d, prob, color=color)

# def plot_first_positives_lines(r, age_separator=(40,60)):
#     db = r.dataset_test_first_diag
#     pos_mask = db.labels == 1
#     pred = np.stack([db.predictions_per_age[age][pos_mask] for age in range(20,80,5)], axis=0)
#     age_below = db.df.AGE.values[pos_mask] < age_separator[0]  
#     age_above = db.df.AGE.values[pos_mask] >= age_separator[1] 
    

#     neg = r.negatives_older60
#     pred_below = [np.array(x) for x in pred[:,age_below]]
#     pred_below = [x for x in pred[:,age_below]]
#     pred_above = [np.array(x) for x in pred[:,age_above]]

#     # color_boxplot = sns.color_palette("copper_r", n_colors=14)
#     color_boxplot = sns.color_palette("BrBG_r", n_colors=12)

#     # bandwidth(pred_below_50[4])

#     # fig, (ax0, ax1, ax2) = plt.figure(figsize=(15,5), dpi=300)
#     fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, sharey=False, figsize=(18,6))
#     # ax0 = plt.subplot(131)
#     for i in range(12):
#         kde0 = _kde_plot(pred_below[i], ax0, color=color_boxplot[i])
#     ax0.set_title(f"Diabetics\n(first diagnostic before {age_separator[0]} years)")
#     ax0.set_ylabel("Density", fontsize=15)
#     ax0.set_xlabel("Risk score", fontsize=15)
#     ax0.set_xlim(-0.01, 1.01)
#     ax0.set_ylim(-0.0001, 2.5)

#     # ax1.subplot(132)
#     for i in range(12):
#         kde1 = _kde_plot(pred_above[i], ax1, color=color_boxplot[i])
#     ax1.set_title(f"Diabetics\n(first diagnostic after {age_separator[1]} years)")
#     ax1.set_ylabel("Density", fontsize=15)
#     ax1.set_xlabel("Risk score", fontsize=15)
#     ax1.set_xlim(-0.01, 1.01)
#     ax1.set_ylim(-0.0001, 2.5)
    
#     # ax2.subplot(133)
#     for i in range(12):
#         kde2 = _kde_plot(neg[0][i], ax2, color=color_boxplot[i])
#     ax2.set_title("Non-diabetics\n(with 60 years or more)")
#     ax2.set_ylabel("Density", fontsize=15)
#     ax2.set_xlabel("Risk score", fontsize=15)
#     ax2.set_xlim(-0.01, 1.01)
#     ax2.set_ylim(-0.0001, 2.5)
#     fig.tight_layout(pad=1)
 
#     cmap = mpl.colors.ListedColormap([color_boxplot[i] for i in range(12)])
#     norm = mpl.colors.Normalize(vmin=17.5, vmax=77.5)
#     cbax = fig.add_axes([1.0, 0.13, 0.02, 0.70]) 
#     fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#                 cax=cbax, orientation='vertical',ticks=np.arange(20,80,5))
#     cbax.set_ylabel('Age (model input)', fontsize=15)

#     return fig

# def plot_first_positives_boxplot(r, age_separator=(40,60)):
#     db = r.dataset_test_first_diag
#     pos_mask = db.labels == 1
#     pred = np.stack([db.predictions_per_age[age][pos_mask] for age in range(20,80,5)], axis=0)
#     age_below = db.df.AGE.values[pos_mask] < age_separator[0]
#     age_above = db.df.AGE.values[pos_mask] >= age_separator[1]
    

#     neg = r.negatives_older60
#     pred_below = [np.array(x) for x in pred[:,age_below]]
#     pred_below = [x for x in pred[:,age_below]]
#     pred_above = [np.array(x) for x in pred[:,age_above]]

#     color_boxplot = sns.color_palette("Spectral_r", n_colors=20)
#     # color_boxplot = sns.color_palette("cool", n_colors=20)

#     fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, sharey=False, figsize=(18,6))
#     # box 01
#     ax0.boxplot(
#             pred_below,
#             showfliers=False,
#             patch_artist=True,
#             labels=[i for i in neg[1]],
#             medianprops=dict(linewidth=2.5, color="black"),
#     )
#     colors = [color_boxplot[int(np.mean(a) * 20)] for a in pred_below]
#     for (i,box) in enumerate(ax0.artists):
#         box.set_facecolor(colors[i])
#     # ax0.set_title("positives\n(age < 50 )")
#     ax0.set_title(f"Diabetics\n(first diagnostic before {age_separator[0]} years)")
#     ax0.set_ylabel("Risk score", fontsize=15)
#     ax0.set_xlabel("Age (model input)", fontsize=15)
#     ax0.set_ylim(0, 1)

#     ax1.boxplot(
#             pred_above,
#             showfliers=False,
#             patch_artist=True,
#             labels=[i for i in neg[1]],
#             medianprops=dict(linewidth=2.5, color="black"),
#     )
#     colors = [color_boxplot[int(np.mean(a) * 20)] for a in pred_above]
#     for (i,box) in enumerate(ax1.artists):
#         box.set_facecolor(colors[i])
#     # ax1.set_title("positives\n(age >= 50 )")
#     ax1.set_title(f"Diabetics\n(first diagnostic after {age_separator[1]} years)")
#     ax1.set_ylabel("Risk score", fontsize=15)
#     ax1.set_xlabel("Age (model input)", fontsize=15)
#     ax1.set_ylim(0, 1)

#     ax2.boxplot(
#             neg[0],
#             showfliers=False,
#             patch_artist=True,
#             labels=[i for i in neg[1]],
#             medianprops=dict(linewidth=2.5, color="black"),
#     )
#     colors = [color_boxplot[int(np.mean(a) * 20)] for a in neg[0]]
#     for (i,box) in enumerate(ax2.artists):
#         box.set_facecolor(colors[i])
#     # ax2.set_title("negatives\n(age >= 60)")
#     ax2.set_title("Non-diabetics\n(with 60 years or more)")
#     ax2.set_ylabel("Risk score", fontsize=15)
#     ax2.set_xlabel("Age (model input)", fontsize=15)
#     ax2.set_ylim(0, 1)

#     fig.tight_layout(pad=1)

#     cmap = mpl.colors.ListedColormap([color_boxplot[i] for i in range(20)])
#     norm = mpl.colors.Normalize(vmin=0, vmax=1)
#     cbax = fig.add_axes([1.0, 0.13, 0.02, 0.70]) 
#     fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#                 cax=cbax, orientation='vertical', ticks=np.arange(0,1.01,.1))
#     # cbax.tick_params(labelsize=15)
#     cbax.set_ylabel(r'$\mathbb{E}$ [risk score]', fontsize=15)
    
#     return fig

# def plot_core_family(r):
#     db = r.dataset_test_unique
#     df = db.df.set_index("id")
#     # print(df)
#     df_fathers = df[df.index.isin(df.fa)]
#     df_mothers = df[df.index.isin(df.mo)]
#     df_children = df[
#         (df.fa.isin(df_fathers.index)) & (df.mo.isin(df_mothers.index))
#     ]
#     # print(df_children)
#     parents = df_children[["fa", "mo"]].groupby(["fa", "mo"]).count().index
#     predictions = np.stack(
#         [db.predictions_per_age[age] for age in range(20, 80, 5)], axis=1
#     )
#     patient_ages = np.array(db.features[:, db._feat_names.index("AGE")])
    
#     plt.figure(figsize=(16, 36), dpi=150)
#     for i in range(len(parents)):
#         ax1 = plt.subplot(9, 4, i + 1)
#         plt.title("father {} mother {}".format(parents[i][0], parents[i][1]))
#         father_mask = (db.df["id"] == parents[i][0]).values
#         mother_mask = (db.df["id"] == parents[i][1]).values
#         pos_mask = db.labels == 1
#         neg_mask = db.labels == 0
#         children_mask = np.logical_and(
#             (db.df["fa"] == parents[i][0]).values,
#             (db.df["mo"] == parents[i][1]).values,
#         )

#         m_pos_fa = np.logical_and(pos_mask, father_mask)
#         m_pos_mo = np.logical_and(pos_mask, mother_mask)
#         m_pos_chi = np.logical_and(pos_mask, children_mask)
#         m_neg_fa = np.logical_and(neg_mask, father_mask)
#         m_neg_mo = np.logical_and(neg_mask, mother_mask)
#         m_neg_chi = np.logical_and(neg_mask, children_mask)

#         if True in m_pos_fa:
#             plt.plot(np.transpose(predictions[m_pos_fa, :]), color=COLORS[9])
#             plt.scatter(
#                 (patient_ages[father_mask] - 20) / 5,
#                 db.predictions[father_mask],
#                 color=COLORS[9],
#             )
#         if True in m_pos_mo:
#             plt.plot(np.transpose(predictions[m_pos_mo, :]), color=COLORS[6])
#             plt.scatter(
#                 (patient_ages[mother_mask] - 20) / 5,
#                 db.predictions[mother_mask],
#                 color=COLORS[6],
#             )
#         if True in m_pos_chi:
#             plt.plot(np.transpose(predictions[m_pos_chi, :]), color=COLORS[7])
#             plt.scatter(
#                 (patient_ages[children_mask] - 20) / 5,
#                 db.predictions[children_mask],
#                 color=COLORS[7],
#             )
#         if True in m_neg_fa:
#             plt.plot(np.transpose(predictions[m_neg_fa, :]), color=COLORS[9], ls="--")
#             plt.scatter(
#                 (patient_ages[father_mask] - 20) / 5,
#                 db.predictions[father_mask],
#                 color=COLORS[9],
#             )
#         if True in m_neg_mo:
#             plt.plot(np.transpose(predictions[m_neg_mo, :]), color=COLORS[6], ls="--")
#             plt.scatter(
#                 (patient_ages[mother_mask] - 20) / 5,
#                 db.predictions[mother_mask],
#                 color=COLORS[6],
#             )
#         if True in m_neg_chi:
#             plt.plot(
#                 np.transpose(predictions[m_neg_chi, :]), color=COLORS[7], ls="--"
#             )
#             plt.scatter(
#                 (patient_ages[children_mask] - 20) / 5,
#                 db.predictions[children_mask],
#                 color=COLORS[7],
#             )
#         # plt.scatter(
#         #     (patient_ages[father_mask] - 20) / 5,
#         #     db.predictions[father_mask],
#         #     c=COLORS[9],
#         # )
#         # plt.scatter(
#         #     (patient_ages[mother_mask] - 20) / 5,
#         #     db.predictions[mother_mask],
#         #     c=COLORS[6],
#         # )
#         # plt.scatter(
#         #     (patient_ages[children_mask] - 20) / 5,
#         #     db.predictions[children_mask],
#         #     c=COLORS[7],
#         # )
#         plt.ylim(0, 1)
#         plt.xticks(range(12), np.arange(20, 80, 5))
#     plt.tight_layout(pad=1)
    

# def plot_family(r):
#         db = r.dataset_test_unique
#         df_pedigree = pd.read_csv("../data/datasets/pedigree.csv")
#         df_pedigree = df_pedigree.set_index("id")

#         ids = db.df[["id", "AGE", "sex", "mo", "fa", "mo_t2d", "fa_t2d", "T2D"]]
#         ids = ids.join(df_pedigree[["famid"]], how="left", on="id")

#         color_by_family = ids["famid"].values
#         famid_sorted = np.unique(np.sort(color_by_family))
#         fam_masks = [color_by_family == i for i in famid_sorted]
#         patient_ages = np.array(db.features[:, db._feat_names.index("AGE")])

#         plt.figure(figsize=(16, 32), dpi=150)
#         for i in range(len(fam_masks)):
#             ax1 = plt.subplot(8, 4, i + 1)
#             p = np.stack(
#                 [db.predictions_per_age[age][fam_masks[i]] for age in range(20, 80, 5)],
#                 axis=0,
#             )
#             m_p = db.labels[fam_masks[i]] == 1
#             m_n = db.labels[fam_masks[i]] == 0

#             pa = patient_ages[fam_masks[i]]
#             pp = db.predictions[fam_masks[i]]

#             if True in m_n:
#                 plt.plot(p[:, m_n], color="grey")
#             if True in m_p:
#                 plt.plot(p[:, m_p], color="red")
#             # print patient age
#             plt.scatter((pa[m_p] - 20) / 5, pp[m_p], c="black", marker="^")
#             plt.scatter((pa[m_n] - 20) / 5, pp[m_n], c="blue", marker="+")

#             plt.title("famid {}".format(famid_sorted[i]))
#             plt.xticks(range(12), np.arange(20, 80, 5))
#             plt.ylim(0, 1)

#         plt.tight_layout(pad=1)
#             # break

#         # if fig_path != "":
#         #     plt.tight_layout(pad=1)
#         #     plt.savefig(fig_path)

#         # plt.show()

# # calibration
# def plot_calibration_line(r):
#     fig = plt.figure(figsize=(6, 6), dpi=300)

#     # Create subplot
#     ax = fig.add_subplot(111)

#     # Plot calibration curve
#     fraction_of_positives, mean_predicted_value = calibration_curve(
#         r.dataset_test_unique.labels, 
#         r.dataset_test_unique.predictions, 
#         n_bins=10, 
#         strategy='quantile'
#     )

#     # Plot perfectly calibrated curve
#     ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

#     # Plot DiabNet curve
#     ax.plot(mean_predicted_value, fraction_of_positives, label="Uncalibrated DiabNet")

#     # Get ECE
#     ece = r.ece(bootnum=10000, interval='HDI')

#     # Get MCE
#     mce = r.mce(bootnum=10000, interval='HDI')

#     # Get Brier Score
#     brier = r.brier(bootnum=10000, interval='HDI')

#     # # ECE and MCE legend
#     ECE = mpatches.Patch(color=COLORS[1], label='ECE = {:.2f}%'.format(ece['value']*100))
#     MCE = mpatches.Patch(color=COLORS[2], label='MCE = {:.2f}%'.format(mce['value']*100))
#     BRIER = mpatches.Patch(color=COLORS[3], label='Brier Score = {:.2f}'.format(brier['value']))

#     # Configuring plot
#     ax.set_title('Calibration curve (Reliability curve)')
#     ax.set_ylabel('Fraction of positives')
#     ax.set_xlabel('Mean predicted value')

#     # Configure legend
#     legend1 = plt.legend(
#         title='Calibration curves',
#         loc="upper left",
#     )

#     legend2 = plt.legend(
#         loc="lower right",
#         handles=[ECE, MCE, BRIER]
#     )
#     ax.add_artist(legend1)
#     ax.add_artist(legend2)

#     # Display plot
#     plt.tight_layout()
#     plt.show()

# def ece_mce_t_p(preds, targets, bins=10):
#     from astropy.stats import bootstrap
#     lower_bound = np.arange(0.0, 1.0, 1.0/bins)
#     upper_bound = lower_bound + 1.0/bins
    
#     ece = np.zeros(1)
#     mce = np.zeros(1)
#     t = np.zeros(bins)
#     p = np.zeros(bins)
#     interval_min = np.zeros(bins)
#     interval_max = np.zeros(bins)
    
    
    
#     for i in range(bins):
#         mask = (preds > lower_bound[i]) * (preds <= upper_bound[i])
#         if np.any(mask):
# #             print(lower_bound[i], upper_bound[i])
# #             print(preds[mask])
# #             print(targets[mask])
#             t[i] = np.mean(targets[mask])
#             p[i] = np.mean(preds[mask])
#             # num of bootstraps is equal to  
# #             data = bootstrap(targets[mask], bootnum=10000, bootfunc=lambda x: np.mean(x) - t[i])
#             data = bootstrap(targets[mask], bootnum=10000, bootfunc=lambda x: np.mean(x))
#             data = np.sort(data)
# #             np.sort(data)
#             #print(st.sem(data))
#             #interval_min[i], interval_max[i] = st.t.interval(alpha=0.95, df=len(data)-1, loc=t[i], scale=st.sem(data))
# #             t[i], _, _, ci = jackknife_stats(targets[mask], np.mean, 0.95)
# #             interval_min[i], interval_max[i] = data[249], data[9749]
#             # interval_min[i], interval_max[i] = data[249]-t[i], data[9749]-t[i]
#             interval_min[i], interval_max[i] = data[499]-t[i], data[9500]-t[i]
#             delta = np.abs(np.mean(preds[mask]) - np.mean(targets[mask]))
#             ece += delta * np.mean(mask)
#             mce = np.maximum(mce, delta)
# #             print(ece, mce)
            
# #     print(interval_min)
# #     print(interval_max)
#     return ece, mce, t, p, np.stack([np.abs(interval_min), interval_max])

# def plot_calibration(r):
#     BINS = 5
#     ece, mce, t, p, ece_ci = ece_mce_t_p(r.dataset_test_unique.predictions, r.dataset_test_unique.labels, bins=BINS)
#     # fig = plt.figure(figsize=(6, 6), dpi=300)
#     fig = plt.figure(figsize=(8,8))


#     # Create subplot
#     ax = fig.add_subplot(111)

#     # Plot calibration curve
#     fraction_of_positives, mean_predicted_value = calibration_curve(
#         r.dataset_test_unique.labels, 
#         r.dataset_test_unique.predictions, 
#         n_bins=5, 
#         strategy='quantile'
#     )
#     # print(fraction_of_positives)
#     # print(mean_predicted_value)

#     # Plot perfectly calibrated curve
#     ax.plot([0, 1], [0, 1], color=COLORS[7], linewidth=1, linestyle='dashed', label="Perfect calibration")

#     # Plot DiabNet curve
#     width = 1.0/BINS
#     xs = np.arange(width/2,1,width)
#     ax.bar(xs, t ,width, yerr=ece_ci, align='center', linewidth=1, alpha=0.99, edgecolor='k', color=COLORS[8], capsize=5.0, ecolor='k', error_kw=dict(mew=3, mfc='g'), label='Observed' )
#     #gap
#     ax.bar(xs, t-xs, width, xs, align='center', linewidth=0.5, alpha=1.0, edgecolor=COLORS[0], hatch='////', fill=False, label='Gap')
#     # ax.bar(mean_predicted_value, fra, width, xs, align='center', linewidth=0.1, alpha=0.5, color='red')
    
#     # ax.plot(mean_predicted_value, fraction_of_positives, label="Uncalibrated DiabNet")
    

#     # Get ECE
#     ece = r.ece(bootnum=10000, interval='HDI')['value']

#     # Get MCE
#     mce = r.mce(bootnum=10000, interval='HDI')['value']

#     # Get Brier Score
#     brier = r.brier(bootnum=10000, interval='HDI')['value']
#     ax.text(0.61, 0.03, f'ECE = {ece:.3f}\nMCE = {mce:.3f}\nBrier Score = {brier:.3f}', c='white', fontsize=18, horizontalalignment='left', bbox=dict(facecolor='k', ec='k'))

#     # # # ECE and MCE legend
#     # ECE = mpatches.Patch(color=COLORS[1], label='ECE = {:.2f}%'.format(ece['value']*100))
#     # MCE = mpatches.Patch(color=COLORS[2], label='MCE = {:.2f}%'.format(mce['value']*100))
#     # BRIER = mpatches.Patch(color=COLORS[3], label='Brier Score = {:.2f}'.format(brier['value']))

#     # Configuring plot
#     ax.set_title('Reliability diagram')
#     ax.set_ylabel('Fraction of positives')
#     ax.set_xlabel('Mean predicted value (Confidence)')

#     ax.set_ylim(0,1)
#     ax.set_xlim(0,1)
#     ax.set_yticks(np.arange(0,1.01, 0.1))
#     ax.set_xticks(np.arange(0,1.01, 0.1))

#     # # Configure legend
#     # legend1 = plt.legend(
#     #     title='Calibration curves',
#     #     loc="upper left",
#     # )

#     # legend2 = plt.legend(
#     #     loc="lower right",
#     #     handles=[ECE, MCE, BRIER]
#     # )
#     # ax.add_artist(legend1)
#     # ax.add_artist(legend2)
#     ax.legend(fontsize=18)

#     # Display plot
#     fig.tight_layout()
#     # plt.show()
#     return fig

# --------------------------------------------------------------------------- #
# -------------- Notebook 4: Family Cross Validation analysis --------------- #
# from diabnet.ensemble import Ensemble
# from diabnet.analysis.report import DiabNetReport
# from matplotlib.lines import Line2D
# from matplotlib.patches import Patch

# COLORS_F = sns.color_palette("Set3")

# def _data2barplot(
#     data: pd.DataFrame, selection: Optional[List[str]] = None, orient: str = "index"
# ) -> Tuple[Dict[str, List[float]], List[str]]:
#     """Prepara pandas.DataFrame to barplot.

#     Parameters
#     ----------
#     data : pd.DataFrame
#         Data of family cross-validation analysis.
#     selection : List[str], optional
#         A list of metrics to be ploted. I must only contain
#         header in data, by default None.
#     orient : str {'dict', 'index'}, optional
#         Determines the type of the values of the dictionary, by default `index`.

#             * 'dict' : dict like {column -> {index -> value}}

#             * 'index' (default) : dict like {index -> {column -> value}}

#     Returns
#     -------
#     data : Dict[str, List[float]]
#         Data prepared to barplot function.
#     labels : List[str]
#         A list of labels to barplot function.

#     Raises
#     ------
#     TypeError
#         "`data` must be a pandas.DataFrame.
#     ValueError
#         `selection` must be columns of `data`.
#     TypeError
#         `orient` must be `dict` or `index`.
#     ValueError
#         `orient` must be `dict` or `index`.
#     """
#     # Check arguments
#     if type(data) not in [pd.DataFrame]:
#         raise TypeError("`data` must be a pandas.DataFrame.")
#     if selection is not None:
#         if type(selection) not in [list]:
#             raise TypeError("`selection` must be a list.")
#         elif not set(selection).issubset(list(data.columns)):
#             raise ValueError("`selection` must be columns of `data`.")
#         # Apply selection
#         data = data[selection]
#     if type(orient) not in [str]:
#         raise TypeError("`orient` must be `dict` or `index`.")
#     elif orient not in ["dict", "index"]:
#         raise ValueError("`orient` must be `dict` or `index`.")

#     # Convert data to dict
#     tmp = data.to_dict(orient)

#     # Get labels
#     labels = list(tmp[list(tmp)[0]].keys())

#     # Prepare data for barplot
#     data = {key: list(tmp[key].values()) for key in tmp.keys()}

#     return data, labels


# def barplot(
#     ax,
#     df,
#     selection: Optional[List[str]] = None,
#     labels: Optional[List[str]] = None,
#     rotation: int = 0,
#     ha: str = "center",
#     colors: Optional[Union[List[float], seaborn.palettes._ColorPalette]] = None,
#     total_width: float = 0.9,
#     single_width: float = 1,
#     legend: bool = True,
# ):
#     """Draws a bar plot with multiple bars per data point.

#     Parameters
#     ----------
#     ax : matplotlib.pyplot.axis
#         The axis we want to draw our plot on.
#     df : Dataframe
#     labels : List[str], optional
#         A list of labels to use as axis ticks. If we want to remove xticks, set
#         to []. If None, use labels from _data2barplot, by default None.
#     rotation : int, optional
#         Rotation (degrees) of xticks, by default 0.
#     ha : str {'left', 'center', 'right'}, optional
#         Horizontal aligments of xticks, by default `center`.
#     colors : Union[List[float], seaborn.palettes._ColorPalette], optional
#         A list of colors which are used for the bars. If None, the colors
#         will be the standard matplotlib color cylel, by default None.
#     total_width : float, optional
#         The width of a bar group. 0.9 means that 90% of the x-axis is covered
#         by bars and 10% will be spaces between the bars, by default 0.9.
#     single_width: float, optional
#         The relative width of a single bar within a group. 1 means the bars
#         will touch eachother within a group, values less than 1 will make
#         these bars thinner, by default 1.
#     legend: bool, optional
#         If this is set to true, a legend will be added to the axis, by default True.
#     """
#     # Number of bars per group
#     n_bars = df.count()[0]

#     colors_by_fam = {fam: colors[i] for i, fam in enumerate(df.sort_index().index)}

#     # The width of a single bar
#     bar_width = total_width / (n_bars+1) # bar_width = 0.9/10

#     # List containing handles for the drawn bars, used for the legend
#     bars = []
    
#     for i, colname in enumerate(selection):
#         df_tmp = df.sort_values(by=colname)
#         x_offset = i
#         c = [colors_by_fam[f] for f in df_tmp.index]
#         for x, y in enumerate(df_tmp[colname].values):
#             bar = ax.bar(
#                 x * bar_width + x_offset - (bar_width * n_bars/2 - bar_width/2),
#                 y,
#                 width=bar_width * single_width,
#                 color=c[x],
#                 edgecolor='gainsboro',
#                 linewidth=1,
#             )
#         bars.append(bar[0])


#     # Set x ticks
#     ax.set_xticks(range(len(labels)), minor=False)
#     ax.set_xticklabels(labels, rotation=rotation, ha=ha)

#     # Draw legend if we need
#     if legend:
#         ax.legend(df.sort_values(by=df.columns[0]).index)

#     # from matplotlib import pyplot as plt

#     # # Check if colors where provided, otherwhise use the default color cycle
#     # if colors is None:
#     #     colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

#     # # Prepare data
#     # if labels is None:
#     #     data, labels = _data2barplot(data, selection=selection, orient=orient)
#     # else:
#     #     data, _ = _data2barplot(data, selection=selection, orient=orient)

#     # # Number of bars per group
#     # n_bars = len(data)

#     # # The width of a single bar
#     # bar_width = total_width / n_bars

#     # # List containing handles for the drawn bars, used for the legend
#     # bars = []

#     # # Iterate over all data
#     # for i, (name, values) in enumerate(data.items()):
#     #     # The offset in x direction of that bar
#     #     x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

#     #     # Draw a bar for every value of that type
#     #     for x, y in enumerate(values):
#     #         bar = ax.bar(
#     #             x + x_offset,
#     #             y,
#     #             width=bar_width * single_width,
#     #             color=colors[i % len(colors)],
#     #         )

#     #     # Add a handle to the last drawn bar, which we'll need for the legend
#     #     bars.append(bar[0])

#     # # Set x ticks
#     # ax.set_xticks(range(len(labels)), minor=False)
#     # ax.set_xticklabels(labels, rotation=rotation, ha=ha)

#     # # Draw legend if we need
#     # if legend:
#     #     ax.legend(bars, data.keys())

# def _get_families_data(r_families):
#     data = {}
#     for fam, r in r_families.items():
#         n = len(r.dataset_test_unique.labels)
#         data[fam] = {
#             "auc": r.auc(bootnum=1000, interval='HDI')['value'],
#             "f1": r.f1(bootnum=1000, interval='HDI')['value'],
#             "acc": r.acc(bootnum=1000, interval='HDI')['value'],
#             "bacc": r.bacc(bootnum=1000, interval='HDI')['value'],
#             "precision": r.precision(bootnum=1000, interval='HDI')['value'],
#             "sensitivity": r.sensitivity(bootnum=1000, interval='HDI')['value'],
#             "specificity": r.specificity(bootnum=1000, interval='HDI')['value'],
#             "avgprec": r.average_precision(bootnum=1000, interval='HDI')['value'],
#             "ece": r.ece(bootnum=10000, interval='HDI')['value'],
#             "mce": r.mce(bootnum=10000, interval='HDI')['value'],
#             "brier": r.brier(bootnum=10000, interval='HDI')['value'],
#             "n": n,
#             "pos%": sum(r.dataset_test_unique.labels)/n,
#         }
#     return pd.DataFrame.from_dict(data, orient='index')

# def _get_families_ages(r_families):
#     ages = {}
#     for fam, r in r_families.items():
#         ages[fam] = r.dataset_test_unique.df.AGE.values
#     return pd.DataFrame.from_dict(ages, orient='index').T

# def _get_families_confusion(r_families):
#     dtmp = pd.DataFrame()
#     for fam, r in r_families.items():
#         tmp = pd.DataFrame(
#         {
#             'age': r.dataset_test_unique.df.AGE.values,
# #             'targets': r.dataset_test_unique.labels,
#             'prediction': r.dataset_test_unique.predictions,
#             'label': r.dataset_test_unique.labels,
#             'famid': len(r.dataset_test_unique.labels) * [fam],
#             'is_correct': len(r.dataset_test_unique.labels) * [True], # init column
#         })
#         dtmp = dtmp.append(tmp)
        
#     def confusion(row):
#         if row['prediction'] > 0.5 and row['label'] > 0.5:
#             return 'TP'
#         elif row['prediction'] <= 0.5 and row['label'] <= 0.5:
#             return 'TN'
#         elif row['prediction'] > 0.5 and row['label'] <= 0.5:
#             return 'FP'
#         elif row['prediction'] <= 0.5 and row['label'] > 0.5:
#             return 'FN'
#     dtmp['confusion'] = dtmp.apply(lambda row: confusion(row), axis=1)
#     dtmp.loc[(dtmp.confusion == 'FP') | (dtmp.confusion == 'FN'),'is_correct'] = False
#     # print(dtmp.where((dtmp.confusion == 'FP') | (dtmp.confusion == 'FN'), other=True)) 
#     return dtmp

# def plot_families_metrics(r_families):
#     # colors = sns.color_palette("Set3")
#     # Create figure
#     # fig, ax = plt.subplots(1, figsize=(20, 6), dpi=300)
#     fig, ax = plt.subplots(1, figsize=(18, 6))

#     data = _get_families_data(r_families)
#     # Barplot metrics
#     barplot(
#         ax,
#         data,
#         selection=[
#             "auc",
#             "avgprec",
#             "bacc",
#             # "acc",
#             "f1",        
#             "precision",
#             "sensitivity",
#             "specificity",
#             "brier",
#             "ece",
#             "mce",
#             "pos%",
#         ],
#         labels=[
#             "AUROC",
#             "AUPRC",
#             "Balanced\nAccuracy",
#             # "Accuracy",
#             "F1-Score", 
#             "Precision",
#             "Sensitivity",
#             "Specificity",
#             "Brier",
#             "ECE",
#             "MCE",
#             "Fraction of\npositives",
#         ],
#         rotation=0,
#         ha="center",
#         colors=COLORS_F,
#         total_width=0.9,
#         single_width=1,
#     )
#     return fig

# def plot_families_ages(r_families):
#     # fig, ax = plt.subplots(1, figsize=(16, 10), dpi=300)
#     fig, ax = plt.subplots(1, figsize=(18, 8))

#     ages = _get_families_ages(r_families)

#     # Plot boxplot for each family  
#     sns.boxplot(
#         data=ages, 
#         palette=COLORS_F, 
#         showmeans=True,
#         meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"red"}
#     )

#     # Configuring plot
#     ax.set_xlabel('Family ID')
#     ax.set_ylabel('Age')

#     # Display plot
#     return fig

# def plot_families_confusion(r_families, with_boxplot=False):
#     # Create figure
#     # fig, ax = plt.subplots(1, figsize=(16, 10), dpi=300)
#     fig, ax = plt.subplots(1, figsize=(18, 10))

#     data = _get_families_confusion(r_families)

#     # Violin plot age
#     sns.stripplot(
#         data=data[data.is_correct], 
#         x='famid', 
#         y='age', 
#         hue='confusion',
#         hue_order=['', 'TN', 'FP', 'FN', 'TP', ''],
#         marker='o',
#         size=8,
#         dodge=True,
#         palette=['white', COLORS_F[3],COLORS_F[2],COLORS_F[1],COLORS_F[0], 'white'], 
#         # palette=COLORS_F,
#         jitter=0.25, 
#         linewidth=1,
#         ax=ax
#     )

#     # Violin plot age
#     sns.stripplot(
#         data=data[~data.is_correct],  
#         x='famid', 
#         y='age', 
#         hue='confusion',
#         hue_order=['', 'TN', 'FP', 'FN', 'TP', ''],
#         marker='X',
#         size=10,
#         dodge=True,
#         # palette=COLORS_F,
#         palette=['white', COLORS_F[3],COLORS_F[2],COLORS_F[1],COLORS_F[0], 'white'], 
#         jitter=0.05, 
#         linewidth=1,
#         ax=ax
#     )

#     if with_boxplot:
#         sns.boxplot(
#             data=data, 
#             x='famid', 
#             y='age',
#             color='gainsboro',
#             width=0.70,
#             boxprops=dict(alpha=.5),
#             linewidth=1,
#             showfliers = False,
#         )

#     # Configuring plot
#     ax.set_xlabel('Family ID', fontsize=18)
#     ax.set_ylabel('Age', fontsize=18)

#     # Configuring legend
#     ax.legend(
#         handles=[
#             Line2D([0], [0], marker='o', color='w', label='True Negative (TN)', markerfacecolor=COLORS_F[3], markersize=10, markeredgecolor='black'),
#             Line2D([0], [0], marker='X', color='w', label='False Positive (FP)', markerfacecolor=COLORS_F[2], markersize=10, markeredgecolor='black'),
#             Line2D([0], [0], marker='X', color='w', label='False Negative (FN)', markerfacecolor=COLORS_F[1], markersize=10, markeredgecolor='black'),
#             Line2D([0], [0], marker='o', color='w', label='True Positive (TP)', markerfacecolor=COLORS_F[0], markersize=10, markeredgecolor='black'),
#         ],
#         loc='upper left',
#         fontsize=15,
#     )
#     return fig

# def plot_correct_by_prob(r_families):
#     # Create figure
#     # fig, ax = plt.subplots(1, figsize=(16, 10), dpi=300)
#     fig, ax = plt.subplots(1, figsize=(18, 10))
#     sns.set_color_codes("colorblind")
#     ax.axhline(0.5, color='k', ls=':')

#     data = _get_families_confusion(r_families)
#     # print(data)
#     # Violin plot
#     ax = sns.violinplot(
#         data=data, 
#         x='famid', 
#         y='prediction', 
#         hue='is_correct',
#         hue_order=[True, False],
#         split=True, 
#         inner='stick',
#         scale='area',
#         linewidth=2,
#     #     palette=colors,
#         palette=['b', 'r'], 
#         saturation=2,

#     )
#     # ax.set

#     plt.setp(ax.collections, alpha=.3, linewidth=1, edgecolor='gainsboro')

#     # Configuring plot
#     ax.set_ylim(-0.01, 1.01)
#     ax.axhline(1.005, color='white', linewidth=3)
#     ax.axhline(-0.005, color='white', linewidth=3)
#     ax.axhline(1.01, color='white')
#     ax.set_xlabel('Family ID', fontsize=18)
#     ax.set_ylabel('Risk score', fontsize=18)
#     # ax.text(9.5, 0.75, 'Positives', rotation='vertical', verticalalignment='center')
#     # ax.text(9.5, 0.25, 'Negatives', rotation='vertical', verticalalignment='center')


#     # Configuring legend
#     ax.legend(
#         handles=[
#             Patch(facecolor='b', edgecolor='black', label='Correct', alpha=0.3),
#             Patch(facecolor='r', edgecolor='black', label='Incorrect', alpha=0.3),
#         ],
#         fontsize='large',
#         title_fontsize='large',
#         title='Predictions',
#         loc='upper left'
#     )
    
#     return fig
# --------------------------------------------------------------------------- #
# ------------- Notebook 5: Examples of predictions to individuals ---------- #

# def _plot_examples(fig, dataset, samples, compact=False, global_grid=None):
#     # fig = plt.figure(figsize=(16,16), dpi=300)
#     # fig = plt.figure(figsize=(18,18))
    
#     if compact:
#         if global_grid != None:
#             outer_grid = global_grid.subgridspec(4, 4, wspace=0., hspace=0.0)
#         else:
#             outer_grid = fig.add_gridspec(4, 4, wspace=0., hspace=0.0)
#     else:
#         outer_grid = fig.add_gridspec(4, 4, wspace=0.3, hspace=0.4)
#     color_h = sns.color_palette("Spectral_r", n_colors=100)
#     for j in range(len(samples)):
#         id = samples[j]
#         label = dataset.df["T2D"].iloc[id]
#         age = dataset.df["AGE"].iloc[id]
#         title = f"patient - id: {dataset.df['id'].iloc[id]}\n(age: {age}, diagnostic: {'P' if label==1 else 'N'})"

#         ages = dataset._age_range
        
#         inner_grid = outer_grid[int(j/4), int(j%4)].subgridspec(1, len(ages), wspace=0.07, hspace=0)

#         i = 0

#         ax_objs = []
#         for i, age in enumerate(ages):
#             # creating new axes object
#             ax_objs.append(fig.add_subplot(inner_grid[0:, i:i+1], zorder=len(ages)-2*i))

#             x = dataset.predictions_per_age_ensemble[age][id]
#             ax_objs[-1].hlines(x, 1, 17, colors=[color_h[int(intensity*99)] for intensity in x], alpha=0.1, linewidth=5)
#             ax_objs[-1].plot(9, np.mean(x), color='k', markersize=5, marker='o')

#             if not compact:
#                 x_d = np.linspace(0,1, 1000)

#                 kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
#                 kde.fit(x[:, None])

#                 logprob = kde.score_samples(x_d[:, None])

                

#                 # plotting the distribution
#                 ax_objs[-1].plot(np.exp(logprob), x_d, color='k',lw=1)

#             if not compact:
#                 # setting uniform x and y lims
#                 ax_objs[-1].set_xlim(1,17)
#                 ax_objs[-1].set_ylim(-0.02,1.02)

#                 # make background transparent
#                 rect = ax_objs[-1].patch
#                 rect.set_alpha(0)

#                 # remove borders, axis ticks, and labels
#                 ax_objs[-1].set_xticklabels([])
#                 ax_objs[-1].set_xticks([10])

#                 if i == 0:
#                     ax_objs[-1].set_ylabel("Risk score", fontsize=15,fontweight="normal")
#                     ax_objs[-1].text(110,-0.20,'Age',fontsize=15,ha="center")
#                     ax_objs[-1].text(110,1.05,title,fontsize=13,ha="center",fontweight="bold")
#                 else:
#                     ax_objs[-1].set_yticklabels([])
#                     ax_objs[-1].set_yticks([])

#                 spines = ["right","left"]
#                 for s in spines:
#                     ax_objs[-1].spines[s].set_visible(False)

#                 ax_objs[-1].text(10,-0.10,age,fontsize=8,ha="center")
            
#             else:
#                 ax_objs[-1].set_xticklabels([])
#                 ax_objs[-1].set_xticks([10])
#                 ax_objs[-1].set_yticklabels([])
#                 # ax_objs[-1].set_yticks([])
#                 ax_objs[-1].set_yticks([0,0.5,1])
#                 ax_objs[-1].set_xlim(1,17)
#                 ax_objs[-1].set_ylim(-0.02,1.02)
#                 spines = ["right","left"]
#                 for s in spines:
#                     ax_objs[-1].spines[s].set_visible(False)
#                 if i == 0: 
#                     ax_objs[-1].spines['left'].set_visible(True)
#                     # if j in [0, 4, 8, 12]:
#                     #     ax_objs[-1].set_ylabel("Risk score", fontsize=15,fontweight="normal")
#                     #     ax_objs[-1].set_yticklabels([0.0,0.5,1.0])
#                     # if j >= 12:
#                     #     ax_objs[-1].text(110,-0.20,'Age',fontsize=15,ha="center")
#                 elif i == 11:
#                     ax_objs[-1].spines['right'].set_visible(True)
#                 # if j >= 12:
#                 #     ax_objs[-1].text(10,-0.10,age,fontsize=8,ha="center")
        
#     return fig

# def plot_elderly_and_positives(r):
#     fig = plt.figure(figsize=(18,18))
#     df = r.dataset_test_first_diag.df
#     N = 16
#     samples = np.random.choice(df[(df.T2D==1)&(df.AGE>60)].index, N, replace=False)
#     fig = _plot_examples(fig, r.dataset_test_first_diag, samples)
#     return fig

# def plot_elderly_and_negatives(r):
#     fig = plt.figure(figsize=(18,18))
#     df = r.dataset_test_unique.df
#     N = 16
#     samples = np.random.choice(df[(df.T2D==0)&(df.AGE>60)].index, N, replace=False)
#     fig = _plot_examples(fig, r.dataset_test_unique, samples)
#     return fig

# def plot_young_and_positives(r):
#     fig = plt.figure(figsize=(18,18))
#     df = r.dataset_test_first_diag.df
#     N = 16
#     samples = np.random.choice(df[(df.T2D==1)&(df.AGE<40)].index, N, replace=False)
#     _plot_examples(fig, r.dataset_test_first_diag, samples)
#     return fig
    
# def plot_young_and_negatives(r, compact=False, subfig=None, global_grid=None):
#     fig = plt.figure(figsize=(18,18))
#     df = r.dataset_test_unique.df
#     N = 16
#     samples = np.random.choice(df[(df.T2D==0)&(df.AGE<40)].index, N, replace=False)
#     if subfig != None:
#         _plot_examples(subfig, r.dataset_test_unique, samples, compact=compact, global_grid=global_grid)
#     else:
#         fig = _plot_examples(fig, r.dataset_test_unique, samples, compact=compact)
#         return fig

# def plot_cases_panel(r):
#     fig = plt.figure(figsize=(18,18))
#     global_grid = fig.add_gridspec(2,2, hspace=0.05, wspace=0.05)
#     # fig.get_constrained_layout_pads
#     # plt.subplots_adjust(hspace=0.0, wspace=0.00)
#     # fig.set_constrained_layout_pads(hspace=0.0, h_pad=0.0, wspace=0.0, w_pad=0.0) 
#     # subfigs = fig.subfigures(2, 2, wspace=0, hspace=0)
#     # plt.subplots_adjust(hspace=0.0, wspace=0.00)
#     plot_young_and_negatives(r, compact=True, subfig=fig, global_grid=global_grid[0,0])
#     plot_young_and_negatives(r, compact=True, subfig=fig, global_grid=global_grid[1,0])
#     plot_young_and_negatives(r, compact=True, subfig=fig, global_grid=global_grid[0,1])
#     plot_young_and_negatives(r, compact=True, subfig=fig, global_grid=global_grid[1,1])
#     return fig

# --------------------------------------------------------------------------- #
# ----------------- Notebook 7: Feature importance analysis ----------------- #


# def get_feature_importance(values, age, sex) -> pd.DataFrame:
#     imp = values.calc_attr(age, sex)
#     # testa se o SNP tem valores 1 ou 2. caso não tenha, sua importancia não pode ser calculada
#     s = {k: [np.mean(imp[k]), np.median(imp[k])] for k in imp if len(imp[k]) > 0}
#     df = pd.DataFrame.from_dict(s, orient="index")
#     df.rename(columns={0: f"{sex}{age}_mean", 1: f"{sex}{age}_median"}, inplace=True)

#     return df


# class ExplainModel:
#     def __init__(self, ensemble, feature_names, dataset_fn):
#         self.ensemble = ensemble
#         self.feat_names = feature_names
#         self.dataset_fn = dataset_fn

#     def encode(self, feat, age=50):
#         feat[self.feat_names.index("AGE")] = age
#         t = torch.Tensor(encode_features(self.feat_names, feat))
#         t.requires_grad_()
#         return t

#     def gen_baselines(self, sex, age=50):
#         baseline = np.zeros(1004, dtype=object)
#         baseline[1001] = sex  # 'M'|'F'|'X'  'X' no information
#         baseline = self.encode(baseline, age)
#         return baseline

#     def gen_inputs(self, feats, age=50):
#         inputs = torch.stack([self.encode(f, age=age) for f in feats])
#         return inputs

#     def calc_attr(self, age, sex):
#         df = pd.read_csv(self.dataset_fn)
#         if sex != "X":
#             df = df[df["sex"] == sex]
#         features = df[self.feat_names].values
#         inputs = self.gen_inputs(features, age=age)
#         baseline = torch.stack(
#             [self.gen_baselines(sex, age=age) for i in range(len(inputs))]
#         )  # same dimension as inputs

#         # attrs = []
#         imp = {self.feat_names[i]: [] for i in range(1000)}
#         # mask to inputs -> snps 1 and 2 == True / snps 0 == False
#         mask = inputs[:, 1, :1000] > 0.5
#         for m in self.ensemble.models:
#             ig = IntegratedGradients(lambda x: m.apply(x))
#             attr = ig.attribute(inputs, baselines=baseline)
#             for i in range(1000):
#                 imp[self.feat_names[i]].append(
#                     attr[:, 1, i][mask[:, i]].detach().numpy()
#                 )

#         for i in range(1000):
#             imp[self.feat_names[i]] = np.concatenate(imp[self.feat_names[i]])

#         return impU

#     def attr_snps_mean(self, attrs, mask):
#         imp = {}
#         for i in range(1000):
#             imp[self.feat_names[i]] = np.concatenate(
#                 [attr[:, 1, i][mask[:, i]].detach().numpy() for attr in attrs]
#             )

#         return imp

# --------------------------------------------------------------------------- #
# ------------- Notebook 8: Relative score ---------- #

# def _distributions(r, age):
#     import numpy as np
#     assert((age >= 20) and (age < 75)) 
#     mask_pos = (r.dataset_test_unique.df.T2D == 1)
#     mask_neg = ((r.dataset_test_unique.df.T2D == 0) & (r.dataset_test_unique.df.AGE > age))
#     if age % 5 != 0:
#         age += 5 - (age % 5)
#     neg_dist = r.dataset_test_unique.predictions_per_age_ensemble[age][mask_neg].flatten()
#     pos_dist = r.dataset_test_unique.predictions_per_age_ensemble[age][mask_pos].flatten()
#     # print(np.max(neg_dist))
#     return neg_dist, pos_dist

# def _relative_score(value, neg, pos):
#     neg_score = np.sum(value < neg)/len(neg)
#     pos_score = np.sum(value > pos)/len(pos)
#     return (neg_score, pos_score)

# def _plot_relative_score(fig, value, neg_dist, pos_dist, title):
#     COLORS = sns.color_palette("colorblind")
#     ax0, ax1 = fig.subplots(1,2, gridspec_kw={'width_ratios': [1, 9]})
#     neg_score, pos_score = _relative_score(value, neg_dist, pos_dist)
#     ax0.bar(0.5, -neg_score, color=COLORS[7])
#     ax0.bar(0.5, pos_score, color=COLORS[3])
#     ax0.set_ylim(-1,1)
#     ax0.set_xticks([])

#     bins_a = int(1 + value * 18) 
#     bins_b = 20 - bins_a
#     # print(bins_a, bins_b)

#     ax1.set_title(title)
#     ax1.hist(neg_dist[neg_dist<value], bins=bins_a, alpha=0.1, color=COLORS[7])
#     ax1.hist(pos_dist[pos_dist<value], bins=bins_a, alpha=0.5, color=COLORS[3])
#     ax1.hist(neg_dist[neg_dist>value], bins=bins_b, alpha=0.5, color=COLORS[7])
#     ax1.hist(pos_dist[pos_dist>value], bins=bins_b, alpha=0.1, color=COLORS[3])


#     ax1.axvline(x=value, color='r', lw=3)   
#     ax1.set_yscale('log')
#     ax1.set_yticks([])
#     ax1.set_xlim(-0.01, 1.01)

# def _plot_scores(dataset, samples, report, pred_age = -1):
#     # fig = plt.figure(figsize=(16,16), dpi=300)
#     fig = plt.figure(figsize=(18,15))

#     subfigs = fig.subfigures(4, 4)

#     for i in range(len(samples)):
#         id = samples[i]
#         label = dataset.df["T2D"].iloc[id]
#         age = dataset.df["AGE"].iloc[id]

#         title = f"patient - id: {dataset.df['id'].iloc[id]}\n(age: {age}, diagnostic: {'P' if label==1 else 'N'})"
#         # subfigs.flat[i].suptitle(title, y=1.0)

#         if pred_age == -1:
#             neg_dist, pos_dist = _distributions(report, age)
#             _plot_relative_score(subfigs.flat[i], dataset.predictions[i], neg_dist, pos_dist, title)
#         else:
#             neg_dist, pos_dist = _distributions(report, pred_age)
#             _plot_relative_score(subfigs.flat[i], dataset.predictions_per_age[pred_age][i], neg_dist, pos_dist, title)

#     return fig


# def plot_relative_score_elderly_and_positives(r, pred_age=-1):
#     df = r.dataset_test_first_diag.df
#     N = 16
#     samples = np.random.choice(df[(df.T2D==1)&(df.AGE>=60)&(df.AGE<75)].index, N, replace=False)
#     fig = _plot_scores(r.dataset_test_first_diag, samples, r, pred_age)
#     # return df[df.index.isin(samples)]
#     return fig

# def plot_relative_score_elderly_and_negatives(r, pred_age=-1):
#     df = r.dataset_test_unique.df
#     N = 16
#     samples = np.random.choice(df[(df.T2D==0)&(df.AGE>=60)&(df.AGE<75)].index, N, replace=False)
#     fig = _plot_scores(r.dataset_test_unique, samples, r, pred_age)
#     # return df[df.index.isin(samples)]
#     return fig

# def plot_relative_score_young_and_positives(r, pred_age=-1):
#     df = r.dataset_test_first_diag.df
#     N = 16
#     samples = np.random.choice(df[(df.T2D==1)&(df.AGE<40)].index, N, replace=False)
#     fig = _plot_scores(r.dataset_test_first_diag, samples, r, pred_age)
#     # return df[df.index.isin(samples)]
#     return fig

# def plot_relative_score_young_and_negatives(r, pred_age=-1):
#     df = r.dataset_test_unique.df
#     N = 16
#     samples = np.random.choice(df[(df.T2D==0)&(df.AGE<40)].index, N, replace=False)
#     fig = _plot_scores(r.dataset_test_unique, samples, r, pred_age)
#     # return df[df.index.isin(samples)]
#     return fig

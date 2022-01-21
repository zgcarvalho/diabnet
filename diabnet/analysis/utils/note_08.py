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


def _distributions(r, age):
    import numpy as np
    assert((age >= 20) and (age < 75)) 
    mask_pos = (r.dataset_test_unique.df.T2D == 1)
    mask_neg = ((r.dataset_test_unique.df.T2D == 0) & (r.dataset_test_unique.df.AGE > age))
    if age % 5 != 0:
        age += 5 - (age % 5)
    neg_dist = r.dataset_test_unique.predictions_per_age_ensemble[age][mask_neg].flatten()
    pos_dist = r.dataset_test_unique.predictions_per_age_ensemble[age][mask_pos].flatten()
    # print(np.max(neg_dist))
    return neg_dist, pos_dist

def _relative_score(value, neg, pos):
    neg_score = np.sum(value < neg)/len(neg)
    pos_score = np.sum(value > pos)/len(pos)
    return (neg_score, pos_score)

def _plot_relative_score(fig, value, neg_dist, pos_dist, title):
    COLORS = sns.color_palette("colorblind")
    ax0, ax1 = fig.subplots(1,2, gridspec_kw={'width_ratios': [1, 9]})
    neg_score, pos_score = _relative_score(value, neg_dist, pos_dist)
    ax0.bar(0.5, -neg_score, color=COLORS[7])
    ax0.bar(0.5, pos_score, color=COLORS[3])
    ax0.set_ylim(-1,1)
    ax0.set_xticks([])

    bins_a = int(1 + value * 18) 
    bins_b = 20 - bins_a
    # print(bins_a, bins_b)

    ax1.set_title(title)
    ax1.hist(neg_dist[neg_dist<value], bins=bins_a, alpha=0.1, color=COLORS[7])
    ax1.hist(pos_dist[pos_dist<value], bins=bins_a, alpha=0.5, color=COLORS[3])
    ax1.hist(neg_dist[neg_dist>value], bins=bins_b, alpha=0.5, color=COLORS[7])
    ax1.hist(pos_dist[pos_dist>value], bins=bins_b, alpha=0.1, color=COLORS[3])


    ax1.axvline(x=value, color='r', lw=3)   
    ax1.set_yscale('log')
    ax1.set_yticks([])
    ax1.set_xlim(-0.01, 1.01)

def _plot_scores(dataset, samples, report, pred_age = -1):
    # fig = plt.figure(figsize=(16,16), dpi=300)
    fig = plt.figure(figsize=(18,15))

    subfigs = fig.subfigures(4, 4)

    for i in range(len(samples)):
        id = samples[i]
        label = dataset.df["T2D"].iloc[id]
        age = dataset.df["AGE"].iloc[id]

        title = f"patient - id: {dataset.df['id'].iloc[id]}\n(age: {age}, diagnostic: {'P' if label==1 else 'N'})"
        # subfigs.flat[i].suptitle(title, y=1.0)

        if pred_age == -1:
            neg_dist, pos_dist = _distributions(report, age)
            _plot_relative_score(subfigs.flat[i], dataset.predictions[i], neg_dist, pos_dist, title)
        else:
            neg_dist, pos_dist = _distributions(report, pred_age)
            _plot_relative_score(subfigs.flat[i], dataset.predictions_per_age[pred_age][i], neg_dist, pos_dist, title)

    return fig


def plot_relative_score_elderly_and_positives(r, pred_age=-1):
    df = r.dataset_test_first_diag.df
    N = 16
    samples = np.random.choice(df[(df.T2D==1)&(df.AGE>=60)&(df.AGE<75)].index, N, replace=False)
    fig = _plot_scores(r.dataset_test_first_diag, samples, r, pred_age)
    # return df[df.index.isin(samples)]
    return fig

def plot_relative_score_elderly_and_negatives(r, pred_age=-1):
    df = r.dataset_test_unique.df
    N = 16
    samples = np.random.choice(df[(df.T2D==0)&(df.AGE>=60)&(df.AGE<75)].index, N, replace=False)
    fig = _plot_scores(r.dataset_test_unique, samples, r, pred_age)
    # return df[df.index.isin(samples)]
    return fig

def plot_relative_score_young_and_positives(r, pred_age=-1):
    df = r.dataset_test_first_diag.df
    N = 16
    samples = np.random.choice(df[(df.T2D==1)&(df.AGE<40)].index, N, replace=False)
    fig = _plot_scores(r.dataset_test_first_diag, samples, r, pred_age)
    # return df[df.index.isin(samples)]
    return fig

def plot_relative_score_young_and_negatives(r, pred_age=-1):
    df = r.dataset_test_unique.df
    N = 16
    samples = np.random.choice(df[(df.T2D==0)&(df.AGE<40)].index, N, replace=False)
    fig = _plot_scores(r.dataset_test_unique, samples, r, pred_age)
    # return df[df.index.isin(samples)]
    return fig

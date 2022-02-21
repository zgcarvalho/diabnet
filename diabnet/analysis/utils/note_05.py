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
from sklearn.neighbors import KernelDensity


COLORS = sns.color_palette("colorblind")

def _plot_examples(fig, dataset, samples, compact=False, global_grid=None):
    # fig = plt.figure(figsize=(16,16), dpi=300)
    # fig = plt.figure(figsize=(18,18))
    
    if compact:
        if global_grid != None:
            outer_grid = global_grid.subgridspec(4, 4, wspace=0., hspace=0.0)
        else:
            outer_grid = fig.add_gridspec(4, 4, wspace=0., hspace=0.0)
    else:
        outer_grid = fig.add_gridspec(4, 4, wspace=0.3, hspace=0.4)
    color_h = sns.color_palette("Spectral_r", n_colors=100)
    for j in range(len(samples)):
        id = samples[j]
        label = dataset.df["T2D"].iloc[id]
        age = dataset.df["AGE"].iloc[id]
        title = f"patient - id: {dataset.df['id'].iloc[id]}\n(age: {age}, diagnostic: {'P' if label==1 else 'N'})"

        ages = dataset._age_range
        
        inner_grid = outer_grid[int(j/4), int(j%4)].subgridspec(1, len(ages), wspace=0.07, hspace=0)

        i = 0

        ax_objs = []
        for i, age in enumerate(ages):
            # creating new axes object
            ax_objs.append(fig.add_subplot(inner_grid[0:, i:i+1], zorder=len(ages)-2*i))

            x = dataset.predictions_per_age_ensemble[age][id]
            ax_objs[-1].hlines(x, 1, 17, colors=[color_h[int(intensity*99)] for intensity in x], alpha=0.1, linewidth=5)
            ax_objs[-1].plot(9, np.mean(x), color='k', markersize=5, marker='o')

            if not compact:
                x_d = np.linspace(0,1, 1000)

                kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
                kde.fit(x[:, None])

                logprob = kde.score_samples(x_d[:, None])

                

                # plotting the distribution
                ax_objs[-1].plot(np.exp(logprob), x_d, color='k',lw=1)

            if not compact:
                # setting uniform x and y lims
                ax_objs[-1].set_xlim(1,17)
                ax_objs[-1].set_ylim(-0.02,1.02)

                # make background transparent
                rect = ax_objs[-1].patch
                rect.set_alpha(0)

                # remove borders, axis ticks, and labels
                ax_objs[-1].set_xticklabels([])
                ax_objs[-1].set_xticks([10])

                if i == 0:
                    ax_objs[-1].set_ylabel("Risk score", fontsize=15,fontweight="normal")
                    ax_objs[-1].text(110,-0.20,'Age',fontsize=15,ha="center")
                    ax_objs[-1].text(110,1.05,title,fontsize=13,ha="center",fontweight="bold")
                else:
                    ax_objs[-1].set_yticklabels([])
                    ax_objs[-1].set_yticks([])

                spines = ["right","left"]
                for s in spines:
                    ax_objs[-1].spines[s].set_visible(False)

                ax_objs[-1].text(10,-0.10,age,fontsize=8,ha="center")
            
            else:
                ax_objs[-1].spines['bottom'].set_color('gray')
                ax_objs[-1].spines['top'].set_color('gray') 
                ax_objs[-1].spines['right'].set_color('gray')
                ax_objs[-1].spines['left'].set_color('gray')
                ax_objs[-1].set_xticklabels([])
                ax_objs[-1].set_xticks([10])
                ax_objs[-1].set_yticklabels([])
                # ax_objs[-1].set_yticks([])
                ax_objs[-1].set_yticks([0,0.5,1])
                ax_objs[-1].set_xlim(1,17)
                ax_objs[-1].set_ylim(-0.05,1.05)
                spines = ["right","left"]
                for s in spines:
                    ax_objs[-1].spines[s].set_visible(False)
                if i == 0: 
                    ax_objs[-1].spines['left'].set_visible(True)
                    # if j == 0:
                    #     ax_objs[-1].text(450, 1.20,'Age',fontsize=15,ha="center")
                    # if j in [0, 4, 8, 12]:
                    #     ax_objs[-1].set_ylabel("Risk score", fontsize=15,fontweight="normal")
                    #     ax_objs[-1].set_yticklabels([0.0,0.5,1.0])
                    # if j >= 12:
                    #     ax_objs[-1].text(110,-0.20,'Age',fontsize=15,ha="center")
                elif i == 11:
                    ax_objs[-1].spines['right'].set_visible(True)
                # if j >= 12:
                #     ax_objs[-1].text(10,-0.10,age,fontsize=8,ha="center")
        
    return fig

def plot_elderly_and_positives(r, compact=False, subfig=None, global_grid=None, sort=False):
    fig = plt.figure(figsize=(18,18))
    df = r.dataset_test_first_diag.df
    N = 16
    samples = np.random.choice(df[(df.T2D==1)&(df.AGE>60)].index, N, replace=False)
    if sort:
        # sort samples by mean risk score 
        mean_samples = np.array([np.mean([r.dataset_test_first_diag.predictions_per_age[age][s] for age in np.arange(20, 75, 5)]) for s in samples])
        samples = samples[np.argsort(mean_samples)]
    if subfig != None:
        _plot_examples(subfig, r.dataset_test_first_diag, samples, compact=compact, global_grid=global_grid)
    else:
        fig = _plot_examples(fig, r.dataset_test_first_diag, samples, compact=compact)
        return fig

def plot_elderly_and_negatives(r, compact=False, subfig=None, global_grid=None, sort=False):
    fig = plt.figure(figsize=(18,18))
    df = r.dataset_test_unique.df
    N = 16
    samples = np.random.choice(df[(df.T2D==0)&(df.AGE>60)].index, N, replace=False)
    if sort:
        # sort samples by mean risk score 
        mean_samples = np.array([np.mean([r.dataset_test_unique.predictions_per_age[age][s] for age in np.arange(20, 75, 5)]) for s in samples])
        samples = samples[np.argsort(mean_samples)]
    if subfig != None:
        _plot_examples(subfig, r.dataset_test_unique, samples, compact=compact, global_grid=global_grid)
    else:
        fig = _plot_examples(fig, r.dataset_test_unique, samples, compact=compact)
        return fig

def plot_young_and_positives(r, compact=False, subfig=None, global_grid=None, sort=False):
    fig = plt.figure(figsize=(18,18))
    df = r.dataset_test_first_diag.df
    N = 16
    samples = np.random.choice(df[(df.T2D==1)&(df.AGE<40)].index, N, replace=False)
    if sort:
        # sort samples by mean risk score 
        mean_samples = np.array([np.mean([r.dataset_test_first_diag.predictions_per_age[age][s] for age in np.arange(20, 75, 5)]) for s in samples])
        samples = samples[np.argsort(mean_samples)]
    if subfig != None:
        _plot_examples(subfig, r.dataset_test_first_diag, samples, compact=compact, global_grid=global_grid)
    else:
        fig = _plot_examples(fig, r.dataset_test_first_diag, samples, compact=compact)
        return fig
    
def plot_young_and_negatives(r, compact=False, subfig=None, global_grid=None, sort=False):
    fig = plt.figure(figsize=(18,18))
    df = r.dataset_test_unique.df
    N = 16
    samples = np.random.choice(df[(df.T2D==0)&(df.AGE<40)].index, N, replace=False)
    if sort:
        # sort samples by mean risk score 
        mean_samples = np.array([np.mean([r.dataset_test_unique.predictions_per_age[age][s] for age in np.arange(20, 75, 5)]) for s in samples])
        samples = samples[np.argsort(mean_samples)]
    if subfig != None:
        _plot_examples(subfig, r.dataset_test_unique, samples, compact=compact, global_grid=global_grid)
    else:
        fig = _plot_examples(fig, r.dataset_test_unique, samples, compact=compact)
        return fig

def plot_cases_panel(r, sort=False):
    fig = plt.figure(figsize=(18,18))
    global_grid = fig.add_gridspec(2,2, hspace=0.05, wspace=0.05)
    # fig.get_constrained_layout_pads
    # plt.subplots_adjust(hspace=0.0, wspace=0.00)
    # fig.set_constrained_layout_pads(hspace=0.0, h_pad=0.0, wspace=0.0, w_pad=0.0) 
    # subfigs = fig.subfigures(2, 2, wspace=0, hspace=0)
    # plt.subplots_adjust(hspace=0.0, wspace=0.00)
    plot_elderly_and_negatives(r, compact=True, subfig=fig, global_grid=global_grid[0,0], sort=sort)
    plot_young_and_negatives(r, compact=True, subfig=fig, global_grid=global_grid[0,1], sort=sort)
    plot_elderly_and_positives(r, compact=True, subfig=fig, global_grid=global_grid[1,0], sort=sort)
    plot_young_and_positives(r, compact=True, subfig=fig, global_grid=global_grid[1,1], sort=sort)
    return fig

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


from diabnet.ensemble import Ensemble
from diabnet.analysis.report import DiabNetReport
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COLORS_F = sns.color_palette("Set3")

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
    df,
    selection: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    rotation: int = 0,
    ha: str = "center",
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
    df : Dataframe
    labels : List[str], optional
        A list of labels to use as axis ticks. If we want to remove xticks, set
        to []. If None, use labels from _data2barplot, by default None.
    rotation : int, optional
        Rotation (degrees) of xticks, by default 0.
    ha : str {'left', 'center', 'right'}, optional
        Horizontal aligments of xticks, by default `center`.
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
    # Number of bars per group
    n_bars = df.count()[0]

    colors_by_fam = {fam: colors[i] for i, fam in enumerate(df.sort_index().index)}

    # The width of a single bar
    bar_width = total_width / (n_bars+1) # bar_width = 0.9/10

    # List containing handles for the drawn bars, used for the legend
    bars = []
    
    for i, colname in enumerate(selection):
        df_tmp = df.sort_values(by=colname)
        x_offset = i
        c = [colors_by_fam[f] for f in df_tmp.index]
        for x, y in enumerate(df_tmp[colname].values):
            bar = ax.bar(
                x * bar_width + x_offset - (bar_width * n_bars/2 - bar_width/2),
                y,
                width=bar_width * single_width,
                color=c[x],
                edgecolor='gainsboro',
                linewidth=1,
            )
        bars.append(bar[0])


    # Set x ticks
    ax.set_xticks(range(len(labels)), minor=False)
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)

    # Draw legend if we need
    if legend:
        ax.legend(df.sort_values(by=df.columns[0]).index)

    # from matplotlib import pyplot as plt

    # # Check if colors where provided, otherwhise use the default color cycle
    # if colors is None:
    #     colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # # Prepare data
    # if labels is None:
    #     data, labels = _data2barplot(data, selection=selection, orient=orient)
    # else:
    #     data, _ = _data2barplot(data, selection=selection, orient=orient)

    # # Number of bars per group
    # n_bars = len(data)

    # # The width of a single bar
    # bar_width = total_width / n_bars

    # # List containing handles for the drawn bars, used for the legend
    # bars = []

    # # Iterate over all data
    # for i, (name, values) in enumerate(data.items()):
    #     # The offset in x direction of that bar
    #     x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

    #     # Draw a bar for every value of that type
    #     for x, y in enumerate(values):
    #         bar = ax.bar(
    #             x + x_offset,
    #             y,
    #             width=bar_width * single_width,
    #             color=colors[i % len(colors)],
    #         )

    #     # Add a handle to the last drawn bar, which we'll need for the legend
    #     bars.append(bar[0])

    # # Set x ticks
    # ax.set_xticks(range(len(labels)), minor=False)
    # ax.set_xticklabels(labels, rotation=rotation, ha=ha)

    # # Draw legend if we need
    # if legend:
    #     ax.legend(bars, data.keys())

def _get_families_data(r_families):
    data = {}
    for fam, r in r_families.items():
        n = len(r.dataset_test_unique.labels)
        data[fam] = {
            "auc": r.auc(bootnum=1000, interval='HDI')['value'],
            "f1": r.f1(bootnum=1000, interval='HDI')['value'],
            "acc": r.acc(bootnum=1000, interval='HDI')['value'],
            "bacc": r.bacc(bootnum=1000, interval='HDI')['value'],
            "precision": r.precision(bootnum=1000, interval='HDI')['value'],
            "sensitivity": r.sensitivity(bootnum=1000, interval='HDI')['value'],
            "specificity": r.specificity(bootnum=1000, interval='HDI')['value'],
            "avgprec": r.average_precision(bootnum=1000, interval='HDI')['value'],
            "ece": r.ece(bootnum=10000, interval='HDI')['value'],
            "mce": r.mce(bootnum=10000, interval='HDI')['value'],
            "brier": r.brier(bootnum=10000, interval='HDI')['value'],
            "n": n,
            "pos%": sum(r.dataset_test_unique.labels)/n,
        }
    return pd.DataFrame.from_dict(data, orient='index')

def _get_families_ages(r_families):
    ages = {}
    for fam, r in r_families.items():
        ages[fam] = r.dataset_test_unique.df.AGE.values
    return pd.DataFrame.from_dict(ages, orient='index').T

def _get_families_confusion(r_families):
    dtmp = pd.DataFrame()
    for fam, r in r_families.items():
        tmp = pd.DataFrame(
        {
            'age': r.dataset_test_unique.df.AGE.values,
#             'targets': r.dataset_test_unique.labels,
            'prediction': r.dataset_test_unique.predictions,
            'label': r.dataset_test_unique.labels,
            'famid': len(r.dataset_test_unique.labels) * [fam],
            'is_correct': len(r.dataset_test_unique.labels) * [True], # init column
        })
        dtmp = dtmp.append(tmp)
        
    def confusion(row):
        if row['prediction'] > 0.5 and row['label'] > 0.5:
            return 'TP'
        elif row['prediction'] <= 0.5 and row['label'] <= 0.5:
            return 'TN'
        elif row['prediction'] > 0.5 and row['label'] <= 0.5:
            return 'FP'
        elif row['prediction'] <= 0.5 and row['label'] > 0.5:
            return 'FN'
    dtmp['confusion'] = dtmp.apply(lambda row: confusion(row), axis=1)
    dtmp.loc[(dtmp.confusion == 'FP') | (dtmp.confusion == 'FN'),'is_correct'] = False
    # print(dtmp.where((dtmp.confusion == 'FP') | (dtmp.confusion == 'FN'), other=True)) 
    return dtmp

def plot_families_metrics(r_families):
    # colors = sns.color_palette("Set3")
    # Create figure
    # fig, ax = plt.subplots(1, figsize=(20, 6), dpi=300)
    fig, ax = plt.subplots(1, figsize=(18, 6))

    data = _get_families_data(r_families)
    # Barplot metrics
    barplot(
        ax,
        data,
        selection=[
            "auc",
            "avgprec",
            "bacc",
            # "acc",
            "f1",        
            "precision",
            "sensitivity",
            "specificity",
            "brier",
            "ece",
            "mce",
            "pos%",
        ],
        labels=[
            "AUROC",
            "AUPRC",
            "Balanced\nAccuracy",
            # "Accuracy",
            "F1-Score", 
            "Precision",
            "Sensitivity",
            "Specificity",
            "Brier",
            "ECE",
            "MCE",
            "Fraction of\npositives",
        ],
        rotation=0,
        ha="center",
        colors=COLORS_F,
        total_width=0.9,
        single_width=1,
    )
    return fig

def plot_families_ages(r_families):
    # fig, ax = plt.subplots(1, figsize=(16, 10), dpi=300)
    fig, ax = plt.subplots(1, figsize=(18, 8))

    ages = _get_families_ages(r_families)

    # Plot boxplot for each family  
    sns.boxplot(
        data=ages, 
        palette=COLORS_F, 
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"red"}
    )

    # Configuring plot
    ax.set_xlabel('Family ID')
    ax.set_ylabel('Age')

    # Display plot
    return fig

def plot_families_confusion(r_families, with_boxplot=False):
    # Create figure
    # fig, ax = plt.subplots(1, figsize=(16, 10), dpi=300)
    fig, ax = plt.subplots(1, figsize=(18, 10))

    data = _get_families_confusion(r_families)

    # Violin plot age
    sns.stripplot(
        data=data[data.is_correct], 
        x='famid', 
        y='age', 
        hue='confusion',
        hue_order=['', 'TN', 'FP', 'FN', 'TP', ''],
        marker='o',
        size=8,
        dodge=True,
        palette=['white', COLORS_F[3],COLORS_F[2],COLORS_F[1],COLORS_F[0], 'white'], 
        # palette=COLORS_F,
        jitter=0.25, 
        linewidth=1,
        ax=ax
    )

    # Violin plot age
    sns.stripplot(
        data=data[~data.is_correct],  
        x='famid', 
        y='age', 
        hue='confusion',
        hue_order=['', 'TN', 'FP', 'FN', 'TP', ''],
        marker='X',
        size=10,
        dodge=True,
        # palette=COLORS_F,
        palette=['white', COLORS_F[3],COLORS_F[2],COLORS_F[1],COLORS_F[0], 'white'], 
        jitter=0.05, 
        linewidth=1,
        ax=ax
    )

    if with_boxplot:
        sns.boxplot(
            data=data, 
            x='famid', 
            y='age',
            color='gainsboro',
            width=0.70,
            boxprops=dict(alpha=.5),
            linewidth=1,
            showfliers = False,
        )

    # Configuring plot
    ax.set_xlabel('Family ID', fontsize=18)
    ax.set_ylabel('Age', fontsize=18)

    # Configuring legend
    ax.legend(
        handles=[
            Line2D([0], [0], marker='o', color='w', label='True Negative (TN)', markerfacecolor=COLORS_F[3], markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='X', color='w', label='False Positive (FP)', markerfacecolor=COLORS_F[2], markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='X', color='w', label='False Negative (FN)', markerfacecolor=COLORS_F[1], markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='True Positive (TP)', markerfacecolor=COLORS_F[0], markersize=10, markeredgecolor='black'),
        ],
        loc='upper left',
        fontsize=15,
    )
    return fig

def plot_correct_by_prob(r_families):
    # Create figure
    # fig, ax = plt.subplots(1, figsize=(16, 10), dpi=300)
    fig, ax = plt.subplots(1, figsize=(18, 10))
    sns.set_color_codes("colorblind")
    ax.axhline(0.5, color='k', ls=':')

    data = _get_families_confusion(r_families)
    # print(data)
    # Violin plot
    ax = sns.violinplot(
        data=data, 
        x='famid', 
        y='prediction', 
        hue='is_correct',
        hue_order=[True, False],
        split=True, 
        inner='stick',
        scale='area',
        linewidth=2,
    #     palette=colors,
        palette=['b', 'r'], 
        saturation=2,

    )
    # ax.set

    plt.setp(ax.collections, alpha=.3, linewidth=1, edgecolor='gainsboro')

    # Configuring plot
    ax.set_ylim(-0.01, 1.01)
    ax.axhline(1.005, color='white', linewidth=3)
    ax.axhline(-0.005, color='white', linewidth=3)
    ax.axhline(1.01, color='white')
    ax.set_xlabel('Family ID', fontsize=18)
    ax.set_ylabel('Risk score', fontsize=18)
    # ax.text(9.5, 0.75, 'Positives', rotation='vertical', verticalalignment='center')
    # ax.text(9.5, 0.25, 'Negatives', rotation='vertical', verticalalignment='center')


    # Configuring legend
    ax.legend(
        handles=[
            Patch(facecolor='b', edgecolor='black', label='Correct', alpha=0.3),
            Patch(facecolor='r', edgecolor='black', label='Incorrect', alpha=0.3),
        ],
        fontsize='large',
        title_fontsize='large',
        title='Predictions',
        loc='upper left'
    )
    
    return fig
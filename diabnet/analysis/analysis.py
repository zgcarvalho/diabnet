import matplotlib
import seaborn
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional

# --------------------------------------------------------------------------- #
# ------------------ Notebook 1: Training results analysis ------------------ #

# TODO: Finish documenting these functions


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
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt

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

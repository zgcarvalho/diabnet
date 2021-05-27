import numpy as np
from typing import Dict, List, Tuple


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
    plt.ylim(0.0, 100.0)
    plt.xlim(-3, 303)

    # Ticks
    plt.yticks(np.arange(0.0, 100, 10), fontsize=9)
    plt.xticks(fontsize=9)

    return ax


# --------------------------------------------------------------------------- #
# ------------------- Notebook 2: Report metrics ensemble ------------------- #

# TODO: Functions

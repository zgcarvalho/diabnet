import os
from tqdm import tqdm
from typing import Dict, Any
from datetime import date
from torch.utils.data import random_split
from diabnet.data import get_feature_names, DiabDataset
from diabnet.train import train

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

CATALOG = {
    "positive": os.path.join(
        DATA_DIR, "visits_sp_unique_train_positivo_1000_random_0.csv"
    ),
    "random": os.path.join(
        DATA_DIR, "visits_sp_unique_train_positivo_0_random_1000.csv"
    ),
    "negative": os.path.join(
        DATA_DIR, "visits_sp_unique_train_positivo_0_negative_1000.csv"
    ),
    "shuffled-snps": os.path.join(
        DATA_DIR, "visits_sp_unique_train_shuffled_snps_positivo_1000_random_0.csv"
    ),
    "shuffled-labels": os.path.join(
        DATA_DIR,
        "visits_sp_unique_train_shuffled_labels_positivo_1000_random_0.csv",
    ),
    "shuffled-ages": os.path.join(
        DATA_DIR, "visits_sp_unique_train_shuffled_ages_positivo_1000_random_0.csv"
    ),
    "shuffled-parents": os.path.join(
        DATA_DIR,
        "visits_sp_unique_train_shuffled_parents_positivo_1000_random_0.csv",
    ),
    "families": {
        fam_id: os.path.join(
            DATA_DIR,
            f"visits_sp_unique_train_famid_{fam_id}_positivo_1000_random_0.csv",
        )
        for fam_id in [0, 10, 14, 1, 30, 32, 33, 3, 43, 7]
    },
}


def train_ensemble(
    fn: str,
    prefix: str,
    log: str,
    params: Dict[str, Dict[str, Any]],
    epochs: int,
    n_ensemble: int,
) -> None:
    """Train an ensemble of DiabNets.

    Parameters
    ----------
    fn : str
        A path to a dataset.
    prefix : str
        A prefix to output files.
    log : str
        A path to a log file.
    params : Dict[str, Dict[str, Any]]
        A dictionary with DiabNet parameters.
    epochs : int
        Number of epochs.
    n_ensemble : int
        Number of models in the emsemble.
    """
    # Open log file
    with open(log, "w") as logfile:
        # Write parameters to log file
        logfile.write(f"PARAMETERS {params}\n")

        # Get feature names
        feat_names = get_feature_names(fn, use_sex=True, use_parents_diagnosis=True)

        # Create DiabNet dataset
        dataset = DiabDataset(
            fn,
            feat_names,
            label_name="T2D",
            soft_label=True,
            soft_label_baseline=params["soft-label-baseline"],
            soft_label_topline=params["soft-label-topline"],
            soft_label_baseline_slope=params["soft-label-baseline-slope"],
        )

        # 10% from training_set to use as validation
        len_trainset = int(0.9 * len(dataset))

        # Train n_ensemble models of DiabNet for epochs
        with tqdm(range(n_ensemble)) as progress:
            for n in progress:
                # Write model number to log file
                logfile.write(f"Model {n:03}\n")

                # Split training and validation sets
                trainset, valset = random_split(
                    dataset, [len_trainset, len(dataset) - len_trainset]
                )

                # Train DiabNet model
                train(
                    params,
                    trainset,
                    valset,
                    epochs,
                    f"{prefix}-{n:03}",
                    logfile,
                    device="cuda",
                    is_trial=False,
                )
                progress.set_description(f"Trained models: [{n+1}/{n_ensemble}]")


def train_from_cfg_file(config: Dict[str, Dict[str, Any]]) -> None:
    """Prepare DiabNet training procedure based on a configuration
    file.

    Parameters
    ----------
    config : Dict[str, Dict[str, Any]]
        A dictionary containing information from a configuration file to
        train DiabNet (parameters, run conditions and datasets).
    """
    # Print configuration file title
    print("[" + ("=" * (len(config["title"]) + 18)) + "]")
    print(f"[======== {config['title']} ========]")

    # Get date
    d = date.today().isoformat()

    # Create results directory
    for dn in ["./results", "./results/models", "./results/logs"]:
        if not os.path.exists(dn):
            os.makedirs(dn)

    # Deserialize configuration file
    parameters = config["params"]
    epochs = config["run"]["epochs"]
    ensemble = config["run"]["ensemble"]
    datasets = config["datasets"]

    # Iterate through datasets
    for dataset in datasets:

        if datasets[dataset]:
            print(f"[==> {dataset}")

            # Prepare output directory
            if not os.path.exists(f"results/models/{dataset}"):
                os.makedirs(f"results/models/{dataset}")

            if dataset == "families":

                for famid, fn in CATALOG[dataset].items():

                    # Create basename for files
                    basename = f"model-famid-{famid}-{parameters['hidden-neurons']}-{parameters['optimizer']}-{parameters['lc-layer']}-{d}"

                    # Prepare prefix
                    if not os.path.exists(f"results/models/{dataset}/famid{famid}"):
                        os.makedirs(f"results/models/{dataset}/famid{famid}")
                    prefix = f"results/models/{dataset}/famid{famid}/{basename}"

                    # Prepare log file
                    log = f"results/logs/{basename}.log"

                    train_ensemble(fn, prefix, log, parameters, epochs, ensemble)

            else:
                # Create basename for files
                basename = f"model-{dataset}-{parameters['hidden-neurons']}-{parameters['optimizer']}-{parameters['lc-layer']}-{d}"

                # Prepare output prefix
                prefix = f"results/models/{dataset}/{basename}"

                # Prepare log file
                log = f"results/logs/{basename}.log"

                # Training DiabNet for dataset
                train_ensemble(
                    CATALOG[dataset], prefix, log, parameters, epochs, ensemble
                )

    print("[" + ("=" * (len(config["title"]) + 18)) + "]")


def main() -> None:
    """Argument parser for training DiabNet with a configuration file."""
    from diabnet import __name__, __version__
    import argparse
    import toml

    # Overrides method in HelpFormatter
    class CapitalisedHelpFormatter(argparse.HelpFormatter):
        def add_usage(self, usage, actions, groups, prefix=None):
            if prefix is None:
                prefix = "Usage: "
            return super(CapitalisedHelpFormatter, self).add_usage(
                usage, actions, groups, prefix
            )

    # argparse
    parser = argparse.ArgumentParser(
        prog="DiabNet",
        description="A Neural Network to predict type 2 diabetes (T2D).",
        formatter_class=CapitalisedHelpFormatter,
        add_help=True,
    )

    # Change parser titles
    parser._positionals.title = "Positional arguments"
    parser._optionals.title = "Optional arguments"

    # Positional arguments
    parser.add_argument(
        "config", help="Path to a configuration file.", default=None, type=str
    )

    # Optional arguments
    parser.add_argument(
        "--version",
        action="version",
        version=f"{__name__} v{__version__}",
        help="Show DiabNet version and exit.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Train DiabNet from file
    train_from_cfg_file(toml.load(args.config))


if __name__ == "__main__":
    main()

from typing import Dict
from datetime import date
import toml
from sys import argv

# from timeit import default_timer
from torch.utils.data import random_split
from diabnet import data
from diabnet.train import train


def net(fn_dataset, fn_out_prefix, fn_log, params, epochs, n_ensemble):
    with open(fn_log, "w") as logfile:
        logfile.write(f"PARAMETERS {params}\n")
        # epochs = 500
        feat_names = data.get_feature_names(
            fn_dataset, use_bmi=False, use_sex=True, use_parents_diagnosis=True
        )
        # print(features)

        dataset = data.DiabDataset(
            fn_dataset,
            feat_names,
            label_name="T2D",
            soft_label=True,
            soft_label_baseline=params["soft-label-baseline"],
            soft_label_topline=params["soft-label-topline"],
        )

        # 10% from training_set to use as validation
        len_trainset = int(0.9 * len(dataset))

        for i in range(n_ensemble):
            # start_time = default_timer()
            print(f"Model {i:03}")
            logfile.write(f"Model {i:03}\n")
            trainset, valset = random_split(
                dataset, [len_trainset, len(dataset) - len_trainset]
            )

            fn_out = f"{fn_out_prefix}-{i:03}.pth"

            train(
                params,
                trainset,
                valset,
                epochs,
                fn_out,
                logfile,
                device="cuda",
                is_trial=False,
            )
            # end_time = default_timer()
            # break
            # break to train only one model
        # print("Elapsed time = ", end_time - start_time)


def train_from(config):
    print(f'Title {config["title"]}')
    params = config["params"]
    d = date.today().isoformat()
    slot = {}

    fn = f'model-{slot}-{config["params"]["hidden-neurons"]}-{config["params"]["optimizer"]}-{config["params"]["lc-layer"]}-{d}'

    if config["datasets"]["positive"]:
        dataset = "data/datasets/visits_sp_unique_train_positivo_1000_random_0.csv"
        fn_out = "data/models/" + fn.format("positive")
        fn_log = "data/logs/" + fn.format("positive")
        net(
            dataset,
            fn_out,
            fn_log,
            config["params"],
            config["run"]["epochs"],
            config["run"]["ensemble"],
        )

    if config["datasets"]["random"]:
        dataset = "data/datasets/visits_sp_unique_train_positivo_0_random_1000.csv"
        fn_out = "data/models/" + fn.format("random")
        fn_log = "data/logs/" + fn.format("random")
        net(
            dataset,
            fn_out,
            fn_log,
            config["params"],
            config["run"]["epochs"],
            config["run"]["ensemble"],
        )

    if config["datasets"]["negative"]:
        dataset = "data/datasets/visits_sp_unique_train_positivo_0_negative_1000.csv"
        fn_out = "data/models/" + fn.format("negative")
        fn_log = "data/logs/" + fn.format("negative")
        net(
            dataset,
            fn_out,
            fn_log,
            config["params"],
            config["run"]["epochs"],
            config["run"]["ensemble"],
        )

    if config["datasets"]["shuffled-positive"]:
        dataset = "data/datasets/visits_sp_unique_train_randomized_1_positivo_1000_random_0.csv"
        fn_out = "data/models/" + fn.format("shuffled-positive")
        fn_log = "data/logs/" + fn.format("shuffled-positive")
        net(
            dataset,
            fn_out,
            fn_log,
            config["params"],
            config["run"]["epochs"],
            config["run"]["ensemble"],
        )

    if config["datasets"]["families"]:

        for fam_id in [0, 10, 14, 1, 30, 32, 33, 3, 43, 7]:
            dataset = f"data/datasets/visits_sp_unique_train_famid_{fam_id}_positivo_1000_random_0.csv"
            fn_out = f'data/models/model-positive-famid-{fam_id}-{config["params"]["hidden-neurons"]}-{config["params"]["optimizer"]}-{config["params"]["lc-layer"]}-{d}'
            fn_log = f'data/logs/model-positive-famid-{fam_id}-{config["params"]["hidden-neurons"]}-{config["params"]["optimizer"]}-{config["params"]["lc-layer"]}-{d}'
            net(
                dataset,
                fn_out,
                fn_log,
                config["params"],
                config["run"]["epochs"],
                config["run"]["ensemble"],
            )


if __name__ == "__main__":
    fn_config = argv[1]
    config = toml.load(fn_config)
    train_from(config)

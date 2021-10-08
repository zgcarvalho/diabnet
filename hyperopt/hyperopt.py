import numpy as np
from scipy import stats
from torch.utils.data import random_split
import numpy as np
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import sys
sys.path.append("../")
from diabnet import data
from diabnet.train import train


def get_trainset(fn_dataset, soft_label_baseline, soft_label_topline, soft_label_baseline_slope):
    feat_names = data.get_feature_names(fn_dataset, use_sex=True, use_parents_diagnosis=True)
    dataset = data.DiabDataset(
            fn_dataset,
            feat_names,
            label_name='T2D',
            soft_label=True,
            soft_label_baseline=soft_label_baseline,
            soft_label_topline=soft_label_topline,
            soft_label_baseline_slope=soft_label_baseline_slope)
    return dataset

def evaluation_function(params):
    epochs = 200
    params["soft-label-baseline-slope"] = -(params["proxy-slope"]-1e-5)/120     
    dataset.adjust_soft_label(params["soft-label-baseline"], 
                           params["soft-label-topline"],
                          params["soft-label-baseline-slope"])
    
#     fn_dataset = '../data/datasets/visits_sp_unique_train_positivo_1000_random_0.csv'
#     dataset = get_trainset(fn_dataset, 
#                            params["soft-label-baseline"], 
#                            params["soft-label-topline"],
#                           params["soft-label-baseline-slope"])

    len_trainset = int(0.8*len(dataset))
    runs = 7
    train_loss = np.zeros(runs)
    val_loss = np.zeros(runs)
    acc = np.zeros(runs)
    bacc = np.zeros(runs)
    fscore = np.zeros(runs)
    ece = np.zeros(runs)
    mce = np.zeros(runs)
    auc = np.zeros(runs)
    avg_prec = np.zeros(runs)
    brier = np.zeros(runs)
    
    for i in range(runs):
        trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])
        train_loss[i], val_loss[i], acc[i], bacc[i], ece[i], mce[i], auc[i], avg_prec[i], fscore[i], brier[i] = train(params, trainset, valset, epochs, is_trial=True, device='cuda')
    
    result = {"train_loss": (train_loss.mean(), stats.sem(train_loss)),
            "val_loss": (val_loss.mean(), stats.sem(val_loss)),
            "acc": (acc.mean(), stats.sem(acc)),
            "bacc": (bacc.mean(), stats.sem(bacc)), 
            "fscore": (fscore.mean(), stats.sem(fscore)),
            "ece": (ece.mean(), stats.sem(ece)),
            "mce": (mce.mean(), stats.sem(mce)),
            "auc": (auc.mean(), stats.sem(auc)),
            "avg_prec": (avg_prec.mean(), stats.sem(avg_prec)),
            "brier": (brier.mean(), stats.sem(brier)),
            "loss_ratio": (val_loss.mean()/train_loss.mean(), 0.0),}
#     print(result)
    return result


if __name__ == '__main__':
    layer = sys.argv[1]

    fn_dataset = '../data/datasets/visits_sp_unique_train_positivo_1000_random_0.csv'
    dataset = get_trainset(fn_dataset, 0.1, 1.0, 0.0)

    parameters=[
        {
            "name": "hidden-neurons",
            "type": "range",
            "bounds": [3, 21],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "dropout",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",
        },
        {
            "name": "lr",
            "type": "range",
            "bounds": [0.001, 0.1],
            "value_type": "float",
            "log_scale": True, 
        },
        {
            "name": "beta1",
            "type": "fixed",
            "value": 0.9,
        },
        {
            "name": "beta2",
            "type": "fixed",
            "value": 0.999,
        },
        {
            "name": "eps",
            "type": "range",
            "bounds": [1e-8, 1e-0],
            "value_type": "float",
            "log_scale": True, 
        },
        {
            "name": "wd",
            "type": "range",
            "bounds": [1e-12, 1e-2],
            "value_type": "float",
            "log_scale": True, 
        },
        {
            "name": "lambda1-dim1",
            "type": "range",
            "bounds": [1e-12, 1e-3],
            "value_type": "float",
            "log_scale": True, 
        },
        {
            "name": "lambda2-dim1",
            "type": "range",
            "bounds": [1e-12, 1e-3],
            "value_type": "float",
            "log_scale": True, 
        },
        {
            "name": "lambda1-dim2",
            "type": "range",
            "bounds": [1e-12, 1e-3],
            "value_type": "float",
            "log_scale": True, 
        },
        {
            "name": "lambda2-dim2",
            "type": "range",
            "bounds": [1e-12, 1e-3],
            "value_type": "float",
            "log_scale": True, 
        },
        {
            "name": "flood-penalty",
            "type": "range",
            "bounds": [0.0, 0.7],
            "value_type": "float",
        },
        {
            "name": "soft-label-topline",
            "type": "range",
            "bounds": [0.51, 0.9999999],
            "value_type": "float",
        },
        {
            "name": "soft-label-baseline",
            "type": "range",
            "bounds": [0.0000001, 0.49],
            "value_type": "float",
        },
        {
            # "name": "soft-label-baseline-slope",
            "name": "proxy-slope",
            "type": "range",
            "bounds": [1e-4, 0.49],
            "value_type": "float",
        },
        {
            "name": "batch-size",
            "type": "fixed",
            "value": 256,
        },
        {
            "name": "sched-steps",
            "type": "range",
            "bounds": [1, 150],
            "value_type": "int",
        },
        {
            "name": "sched-gamma",
            "type": "range",
            "bounds": [0.05, 1.0],
            "value_type": "float",
        },
        {
            "name": "optimizer",
            "type": "fixed",
            "value": "adamw",
        },
        {
            "name": "lc-layer",
            "type": "fixed",
            "value": layer,
        },
    ]
    ax_client = AxClient()
    ax_client.create_experiment(
        name="experiment",
        parameters=parameters,
        objectives={
            # `threshold` arguments are optional
            "bacc": ObjectiveProperties(minimize=False, threshold=0.78), 
            "brier": ObjectiveProperties(minimize=True, threshold=0.11),
        },
        overwrite_existing_experiment=True,
        parameter_constraints=["proxy-slope <= soft-label-baseline"],
        tracking_metric_names=["train_loss","val_loss","acc","bacc", "fscore","ece","mce","auc","avg_prec","brier","loss_ratio"],
        is_test=False,
    )

    for i in range(500):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluation_function(parameters))
    
    ax_client.save_to_json_file(f'hyperopt_20210924_{layer}.json')
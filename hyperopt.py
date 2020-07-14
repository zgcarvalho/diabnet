from diabnet import data
from diabnet.train import train
from torch.utils.data import random_split
import optuna

# DATASET = './datasets/visits_sp_unique_train_positivo_1000_negative_0.csv'

POSITIVE_TRAINSET = None
RANDOM_TRAINSET = None

def get_trainset(fn, soft_label_alpha):
    feat = data.get_feature_names(fn, BMI=False, sex=True, parents_diagnostics=True)
    dataset = data.DiabDataset(fn, feat, random_age=False, soft_label=True, soft_label_alpha=soft_label_alpha)
    return dataset

def objective(trial):
    params = {
        "l1_neurons": trial.suggest_categorical('l1_neurons', [8,13,21]),
        "l2_neurons": 0,
        "l3_neurons": 0,
        "dp0": trial.suggest_discrete_uniform('dp0', 0.35, 0.7, 0.05),
        "dp1": 0.0, # 0.5
        "dp2": 0,
        "dp3": 0,
        "lr": trial.suggest_uniform('lr', 0.001, 0.007),
        "wd": 0.000001,
        "lambda1_dim1": trial.suggest_loguniform('lambda1_dim1', 1e-10, 5e-6),
        "lambda2_dim1": trial.suggest_loguniform('lambda2_dim1', 0.00001, 0.001),
        "lambda1_dim2": trial.suggest_loguniform('lambda1_dim2', 1e-10, 5e-6),
        "lambda2_dim2": trial.suggest_loguniform('lambda2_dim2', 0.00001, 0.01),
        "flood_penalty":trial.suggest_uniform('flood_penalty', 0.0, 0.40),
        "soft_label_alpha": trial.suggest_uniform('soft_label_alpha', 0.0, 0.25)
    }
    epochs = 2500

    global POSITIVE_TRAINSET
    global RANDOM_TRAINSET

    fn_positive_dataset = './datasets/visits_sp_unique_train_positivo_1000_random_0.csv'
    if POSITIVE_TRAINSET == None:
        POSITIVE_TRAINSET = get_trainset(fn_positive_dataset, params["soft_label_alpha"])

    # fn_random_dataset = './datasets/visits_sp_unique_train_randomized_1_positivo_1000_random_0.csv'
    # if RANDOM_TRAINSET == None:
    #     RANDOM_TRAINSET = get_trainset(fn_random_dataset)
    
    len_trainset = int(0.9*len(POSITIVE_TRAINSET))
    positive_trainset, positive_valset = random_split(POSITIVE_TRAINSET, [len_trainset, len(POSITIVE_TRAINSET)-len_trainset])
    # random_trainset, random_valset = random_split(RANDOM_TRAINSET, [len_trainset, len(RANDOM_TRAINSET)-len_trainset])

    positive_train_loss, positive_val_loss = train(params, positive_trainset, positive_valset, epochs, is_trial=True, device='cuda')
    # random_train_loss, random_val_loss = train(params, random_trainset, random_valset, epochs, is_trial=True, device='cuda')
    # score = positive_train_loss - random_train_loss
    score = positive_val_loss
    print(f"Score {score}")
    return score




if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1000)
    
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

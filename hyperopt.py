from diabnet import data
from diabnet.train import train
from torch.utils.data import random_split
import optuna

# DATASET = './datasets/visits_sp_unique_train_positivo_1000_negative_0.csv'

def objective(trial):
    params = {
        "l1_neurons": trial.suggest_categorical('l1_neurons', [3,5,8,13,21,34,55,89,144,233]),
        "l2_neurons": 0,
        "l3_neurons": 0,
        "dp0": trial.suggest_discrete_uniform('dp0', 0.05, 0.8, 0.05),
        "dp1": 0.0, # 0.5
        "dp2": 0,
        "dp3": 0,
        "lr": trial.suggest_loguniform('lr', 0.0001, 0.01),
        "wd": 0.000001,
        "lambda1_dim1": trial.suggest_loguniform('lambda1_dim1', 0.00001, 0.001),
        "lambda2_dim1": trial.suggest_loguniform('lambda2_dim1', 0.001, 0.1),
        "lambda1_dim2": trial.suggest_loguniform('lambda1_dim2', 0.00001, 0.001),
        "lambda2_dim2": trial.suggest_loguniform('lambda2_dim2', 0.001, 0.1),
        "flood_penalty":trial.suggest_uniform('flood_penalty', 0.0, 0.6)
    }
    epochs = 2500

    fn_positive_dataset = './datasets/visits_sp_unique_train_positivo_1000_random_0.csv'
    fn_random_dataset = './datasets/visits_sp_unique_train_randomized_1_positivo_1000_random_0.csv'

    positive_features = data.get_feature_names(fn_positive_dataset, BMI=False, sex=True, parents_diagnostics=True)
    random_features = data.get_feature_names(fn_random_dataset, BMI=False, sex=True, parents_diagnostics=True)

    positive_dataset = data.DiabDataset(fn_positive_dataset, positive_features, random_age=False, soft_label=True)
    random_dataset = data.DiabDataset(fn_random_dataset, random_features, random_age=False, soft_label=True)

    len_trainset = int(0.9*len(positive_dataset))
    positive_trainset, positive_valset = random_split(positive_dataset, [len_trainset, len(positive_dataset)-len_trainset])
    random_trainset, random_valset = random_split(random_dataset, [len_trainset, len(random_dataset)-len_trainset])

    # fn_out = f"diabnet/models/model-32-sp-soft-label-randomized_1_positives-1000-dropout0-05-bn-decay-flood-{i:03}.pth"
    # print("model saved to:", fn_out)
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

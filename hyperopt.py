from diabnet import data
from diabnet.train import train
from torch.utils.data import random_split
import optuna

# DATASET = './datasets/visits_sp_unique_train_positivo_1000_negative_0.csv'

POSITIVE_TRAINSET = None
RANDOM_TRAINSET = None

def get_trainset(fn, soft_label_alpha):
    feat = data.get_feature_names(fn, BMI=False, sex=True, parents_diagnostics=True)
    dataset = data.DiabDataset(fn, feat, soft_label=True, soft_label_alpha=soft_label_alpha)
    return dataset

def objective(trial):
    params = {
        "hidden_neurons": trial.suggest_int('hidden_neurons', 3,13), #13
        "dropout": trial.suggest_uniform('dropout', 0.45, 0.7),
        "lr": trial.suggest_loguniform('lr', 0.0007, 0.07),
        "beta1": 1.0 - trial.suggest_loguniform('beta1', 1e-5, 1e-1),
        "beta2": 1.0 - trial.suggest_loguniform('beta2', 1e-5, 1e-1),
        "eps": trial.suggest_loguniform('eps', 1e-10, 1e-1),
        "wd": 0.000001,
        "lambda1_dim1": trial.suggest_loguniform('lambda1_dim1', 1e-10, 1e-1),
        "lambda2_dim1": trial.suggest_loguniform('lambda2_dim1', 1e-10, 1e-1),
        "lambda1_dim2": trial.suggest_loguniform('lambda1_dim2', 1e-10, 1e-1),
        "lambda2_dim2": trial.suggest_loguniform('lambda2_dim2', 1e-10, 1e-1),
        "flood_penalty": trial.suggest_uniform('flood_penalty', 0.0, 0.20), #0.39
        "soft_label_alpha": trial.suggest_uniform('soft_label_alpha', 0.0, 0.20),
        "batch_size": 256
        
        # # "l1_neurons": trial.suggest_categorical('l1_neurons', [8,13,21]),
        # "batch_size": 256,
        # "l1_neurons": trial.suggest_int('l1_neurons', 3,13),
        # "l2_neurons": 0,
        # "l3_neurons": 0,
        # # "dp0": trial.suggest_discrete_uniform('dp0', 0.35, 0.7, 0.05),
        # "dp0": trial.suggest_uniform('dp0', 0.45, 0.7),
        # "dp1": 0.0, # 0.5
        # "dp2": 0,
        # "dp3": 0,
        # # "lr": 0.004,
        # "lr" : trial.suggest_loguniform('lr', 0.0007, 0.07),
        # "wd": 0.000001,
        # "lambda1_dim1": 0.0,
        # "lambda2_dim1": 0.0,
        # "lambda1_dim2": 0.0,
        # "lambda2_dim2": 0.00025,
        # "flood_penalty":trial.suggest_uniform('flood_penalty', 0.0, 0.20),
        # "soft_label_alpha": trial.suggest_uniform('soft_label_alpha', 0.0, 0.20)
    }
    epochs = 500

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

    train_loss, val_loss, acc, bacc, ece, mce, auc = train(params, positive_trainset, positive_valset, epochs, is_trial=True, device='cuda')
    
    # random_train_loss, random_val_loss = train(params, random_trainset, random_valset, epochs, is_trial=True, device='cuda')
    # score = positive_train_loss - random_train_loss
    score = val_loss
    trial.set_user_attr('train_loss', train_loss)
    trial.set_user_attr('accuracy', acc)
    trial.set_user_attr('balanced_accuracy', bacc)
    trial.set_user_attr('ece', ece)
    trial.set_user_attr('mce', mce)
    trial.set_user_attr('auc', auc)
    print(f"Score {score}")
    return score




if __name__ == "__main__":
    study = optuna.create_study(study_name='radam', direction='minimize', storage='sqlite:///hyperopt_batch256_round0.db', load_if_exists=True)
    study.optimize(objective, n_trials=1000)
    
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

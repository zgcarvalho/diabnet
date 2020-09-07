from diabnet import data
from diabnet.train import train
from torch.utils.data import random_split
import optuna


def get_trainset(fn, soft_label_alpha):
    feat = data.get_feature_names(fn, BMI=False, sex=True, parents_diagnostics=True)
    dataset = data.DiabDataset(fn, feat, soft_label=True, soft_label_alpha=soft_label_alpha)
    return dataset


def objective(trial):
    # round0
    params = {
        "hidden_neurons": trial.suggest_int('hidden_neurons', 3, 7), #13
        "dropout": trial.suggest_uniform('dropout', 0.3, 0.7),
        "lr": trial.suggest_loguniform('lr', 0.001, 0.1),
        "beta1": 0.9,
        "beta2": 0.99,
        "eps": trial.suggest_loguniform('eps', 1e-8, 1e-3),
        "wd": trial.suggest_loguniform('wd', 1e-7, 1e-2),
        "lambda1_dim1": trial.suggest_loguniform('lambda1_dim1', 1e-9, 1e-3),
        "lambda2_dim1": trial.suggest_loguniform('lambda2_dim1', 1e-9, 1e-3),
        "lambda1_dim2": trial.suggest_loguniform('lambda1_dim2', 1e-9, 1e-3),
        "lambda2_dim2": trial.suggest_loguniform('lambda2_dim2', 1e-9, 1e-3),
        "flood_penalty": trial.suggest_uniform('flood_penalty', 0.0, 0.2), #0.39
        "soft_label_alpha": trial.suggest_uniform('soft_label_alpha', 0.0, 0.20),
        "batch_size": 256
    }
    # round 1
    # params = {
    #     "hidden_neurons": 3, #13
    #     "dropout": 0.5,
    #     "lr": 0.027,
    #     "beta1": 0.99985,
    #     "beta2": 0.99999,
    #     "eps": trial.suggest_loguniform('eps', 1e-9, 1e-6),
    #     "lambda1_dim1": trial.suggest_loguniform('lambda1_dim1', 1e-10, 1e-6),
    #     "lambda2_dim1": trial.suggest_loguniform('lambda2_dim1', 1e-10, 1e-6),
    #     "lambda1_dim2": trial.suggest_loguniform('lambda1_dim2', 1e-10, 1e-6),
    #     "lambda2_dim2": trial.suggest_loguniform('lambda2_dim2', 1e-10, 1e-6),
    #     "flood_penalty": trial.suggest_uniform('flood_penalty', 0.0, 0.20), #0.39
    #     "soft_label_alpha": trial.suggest_uniform('soft_label_alpha', 0.0, 0.05),
    #     "batch_size": 256
    # }
    epochs = 1500
        
    fn_dataset = './datasets/visits_sp_unique_train_positivo_1000_random_0.csv'
    dataset = get_trainset(fn_dataset, params["soft_label_alpha"])

    len_trainset = int(0.9*len(dataset))
    trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

    train_loss, val_loss, acc, bacc, ece, mce, auc, avg_prec = train(params, trainset, valset, epochs, is_trial=True, device='cuda')
    
    score = avg_prec
    trial.set_user_attr('train_loss', train_loss)
    trial.set_user_attr('val_loss', val_loss)
    trial.set_user_attr('accuracy', acc)
    trial.set_user_attr('balanced_accuracy', bacc)
    trial.set_user_attr('ece', ece)
    trial.set_user_attr('mce', mce)
    trial.set_user_attr('auc', auc)
    trial.set_user_attr('avg_prec', avg_prec)
    print(f"Score {score}")
    return score




if __name__ == "__main__":
    study = optuna.create_study(study_name='adamw_avg_prec', direction='maximize', storage='sqlite:///hyperopt_batch256_adamw_avg_prec_round0.db', load_if_exists=True)
    study.optimize(objective, n_trials=1000)
    
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

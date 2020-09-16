from diabnet import data
from diabnet.train import train
from torch.utils.data import random_split
from timeit import default_timer
# DATASET = './datasets/visits_sp_unique_train_positivo_1000_negative_0.csv'

def net(fn_dataset, fn_out_prefix, fn_log):
    with open(fn_log, 'w') as logfile:
        params = {
            'hidden_neurons': 6,
            'dropout': 0.08658128211832698,
            'lr': 0.002530315363231387,
            'eps': 1.6516808099564973e-07,
            'wd': 7.141119204788848e-05,
            'lambda1_dim1': 1.669665721023354e-06,
            'lambda2_dim1': 6.368583640194637e-07,
            'lambda1_dim2': 2.0342028053277926e-08,
            'lambda2_dim2': 2.0678941459740793e-11,
            'flood_penalty': 0.18613187374510126,
            'soft_label_alpha': 0.10348253777080188,
            'sched-steps': 22,
            'sched-gamma': 0.6213666892994011,
            'beta1': 0.9,
            'beta2': 0.999,
            'batch_size': 256,
            'opt': 'radam',
            'lc-layer': 'lc2',
        }
        logfile.write(f'PARAMETERS {params}\n')

        epochs = 500

        features = data.get_feature_names(fn_dataset, BMI=False, sex=True, parents_diagnostics=True)
        # print(features)
        
        dataset = data.DiabDataset(fn_dataset, features, soft_label=True, soft_label_alpha=params["soft_label_alpha"])
        len_trainset = int(0.9*len(dataset))
        
        for i in range(100):
            # start_time = default_timer()
            print(f"Model {i:03}")
            logfile.write(f"Model {i:03}\n")
            trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

            fn_out = f"{fn_out_prefix}-{i:03}.pth"

            train(params, trainset, valset, epochs, fn_out, logfile, device='cuda', is_trial=False)
            # end_time = default_timer()
            # break
            # break to train only one model
        # print("Elapsed time = ", end_time - start_time)


if __name__ == "__main__":

    net(fn_dataset='./datasets/visits_sp_unique_train_positivo_1000_random_0.csv',
        fn_out_prefix='diabnet/models/model-6-soft-label-age-positives-1000-dropout0-bn-decay-flood-hyperopt-ax-batch256-adamw-lc1-20200915',
        fn_log='log_model-6-soft-label-age-positives-1000-dropout0-bn-decay-flood-hyperopt-ax-batch256-adamw-lc1-20200915')
   
    net(fn_dataset='./datasets/visits_sp_unique_train_positivo_0_random_1000.csv',
        fn_out_prefix='diabnet/models/model-6-soft-label-age-random-1000-dropout0-bn-decay-flood-hyperopt-ax-batch256-adamw-lc1-20200915',
        fn_log='log_model-6-soft-label-age-random-1000-dropout0-bn-decay-flood-hyperopt-ax-batch256-adamw-lc1-20200915')
  
    net(fn_dataset='./datasets/visits_sp_unique_train_positivo_0_negative_1000.csv',
        fn_out_prefix='diabnet/models/model-6-soft-label-age-negatives-1000-dropout0-bn-decay-flood-hyperopt-ax-batch256-adamw-lc1-20200915',
        fn_log='log_model-6-soft-label-age-negatives-1000-dropout0-bn-decay-flood-hyperopt-ax-batch256-adamw-lc1-20200915')
    
    net(fn_dataset='./datasets/visits_sp_unique_train_randomized_1_positivo_1000_random_0.csv',
        fn_out_prefix='diabnet/models/model-6-soft-label-age-randomized_positives-1000-dropout0-bn-decay-flood-hyperopt-ax-batch256-adamw-lc1-20200915',
        fn_log='log_model-6-soft-label-age-randomized_positives-1000-dropout0-bn-decay-flood-hyperopt-ax-batch256-adamw-lc1-20200915')
 

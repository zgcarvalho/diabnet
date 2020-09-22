from diabnet import data
from diabnet.train import train
from torch.utils.data import random_split
from timeit import default_timer
# DATASET = './datasets/visits_sp_unique_train_positivo_1000_negative_0.csv'

def net(fn_dataset, fn_out_prefix, fn_log):
    with open(fn_log, 'w') as logfile:
        params = {
            'hidden_neurons': 16,
            'dropout': 0.10,
            'lr': 0.005,
            'eps': 2e-05,
            'wd': 6.5e-09,
            'lambda1_dim1': 2.2e-05,
            'lambda2_dim1': 9e-07,
            'lambda1_dim2': 3.5e-08,
            'lambda2_dim2': 1.3e-08,
            'flood_penalty': 0.14,
            'soft_label_alpha': 0.2,
            'sched-steps': 48,
            'sched-gamma': 0.30,
            'beta1': 0.9,
            'beta2': 0.999,
            'batch_size': 256,
            'opt': 'adamw',
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
    neurons = '16'
    optimizer = 'adamw'
    lc = 'lc2'
    date = '20200921'
    slot = {}
    
    DATASET = './datasets/visits_sp_unique_train_positivo_1000_random_0.csv'
    FN_OUT = f'diabnet/models/model-{neurons}-soft-label-age-{slot}-1000-dropout0-bn-decay-flood-hyperopt-ax-batch256-{optimizer}-{lc}-{date}'
    FN_LOG = f'log_model-{neurons}-soft-label-age-{slot}-1000-dropout0-bn-decay-flood-hyperopt-ax-batch256-{optimizer}-{lc}-{date}'

    net(fn_dataset=DATASET,
        fn_out_prefix=FN_OUT.format('positives'),
        fn_log=FN_LOG.format('positives'))
   
    net(fn_dataset=DATASET,
        fn_out_prefix=FN_OUT.format('random'),
        fn_log=FN_LOG.format('random'))
  
    net(fn_dataset=DATASET,
        fn_out_prefix=FN_OUT.format('negatives'),
        fn_log=FN_LOG.format('negatives'))

    net(fn_dataset=DATASET,
        fn_out_prefix=FN_OUT.format('randomized_positives'),
        fn_log=FN_LOG.format('randomized_positives'))
 

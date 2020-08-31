from diabnet import data
from diabnet.train import train
from torch.utils.data import random_split
from timeit import default_timer
# DATASET = './datasets/visits_sp_unique_train_positivo_1000_negative_0.csv'

def net(fn_dataset):
    # neural network to SELECTED features with fixed dropout=0.5 during hyperopt
    for i in range(100):
        # start_time = default_timer()
        print(f"Model {i:03}")
        params = {
            "hidden_neurons": 6, #13
            "dropout": 0.51,
            "lr": 0.027,
            "beta1": 0.9998,
            "beta2": 0.9994,
            "eps": 1.5e-6,
            "wd": 1.2e-5,
            "lambda1_dim1": 1e-6,
            "lambda2_dim1": 3.3e-9,
            "lambda1_dim2": 2e-7,
            "lambda2_dim2": 5e-8,
            "flood_penalty": 0.03, #0.39
            "soft_label_alpha": 0.05,
            "batch_size": 256
        }

        epochs = 2500


        features = data.get_feature_names(fn_dataset, BMI=False, sex=True, parents_diagnostics=True)
        # print(features)
        
        dataset = data.DiabDataset(fn_dataset, features, soft_label=True, soft_label_alpha=params["soft_label_alpha"])
        len_trainset = int(0.9*len(dataset))
        trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

        fn_out = f"diabnet/models/model-6-soft-label-age-positives-2500-dropout0-bn-decay-flood-hyperopt-batch256-lc1-20200831-{i:03}.pth"
        print("model saved to:", fn_out)
        train(params, trainset, valset, epochs, fn_out, device='cuda', is_trial=False)
        # end_time = default_timer()
        # break
        # break to train only one model
    # print("Elapsed time = ", end_time - start_time)


if __name__ == "__main__":

    net('./datasets/visits_sp_unique_train_positivo_1000_random_0.csv')

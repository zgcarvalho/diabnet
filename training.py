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
            "hidden_neurons": 3, #13
            "dropout": 0.4,
            "lr": 0.004,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-7,
            "wd": 0.000001,
            "lambda1_dim1": 0.00000,
            "lambda2_dim1": 0.00000,
            "lambda1_dim2": 0.00001,
            "lambda2_dim2": 0.00025,
            "flood_penalty": 0.05, #0.39
            "soft_label_alpha": 0.09,
            "batch_size": 256
        }
        # params = {
        #     "l1_neurons": 3, #13
        #     "l2_neurons": 0,
        #     "l3_neurons": 0,
        #     "dp0": 0.4,
        #     "dp1": 0.0, # 0.5
        #     "dp2": 0,
        #     "dp3": 0,
        #     "lr": 0.004,
        #     "wd": 0.000001,
        #     "lambda1_dim1": 0.0000,
        #     "lambda2_dim1": 0.0000,
        #     "lambda1_dim2": 0.0000,
        #     "lambda2_dim2": 0.00025,
        #     "flood_penalty": 0.05, #0.39
        #     "soft_label_alpha": 0.09,
        #     "batch_size": 256
        # }

        epochs = 2500


        features = data.get_feature_names(fn_dataset, BMI=False, sex=True, parents_diagnostics=True)
        # print(features)
        
        dataset = data.DiabDataset(fn_dataset, features, soft_label=True, soft_label_alpha=params["soft_label_alpha"])
        len_trainset = int(0.9*len(dataset))
        trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

        fn_out = f"diabnet/models/model-3-soft-label-age-positives-2500-dropout0-bn-decay-flood-hyperopt-batch256-lc1-bias-swa-{i:03}.pth"
        print("model saved to:", fn_out)
        train(params, trainset, valset, epochs, fn_out, device='cuda', is_trial=False)
        # end_time = default_timer()
        # break
        # break to train only one model
    # print("Elapsed time = ", end_time - start_time)


if __name__ == "__main__":

    net('./datasets/visits_sp_unique_train_positivo_1000_random_0.csv')

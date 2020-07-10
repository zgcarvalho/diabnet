from diabnet import data
from diabnet.train import train
from torch.utils.data import random_split

# DATASET = './datasets/visits_sp_unique_train_positivo_1000_negative_0.csv'

def net(fn_dataset):
    # neural network to SELECTED features with fixed dropout=0.5 during hyperopt
    for i in range(100):
        print(f"Model {i:03}")
        params = {
            "l1_neurons": 13, # 256
            "l2_neurons": 0,
            "l3_neurons": 0,
            "dp0": 0.45,
            "dp1": 0.0, # 0.5
            "dp2": 0,
            "dp3": 0,
            "lr": 0.0045,
            "wd": 0.000001,
            "lambda1_dim1": 0.0000,
            "lambda2_dim1": 0.000075,
            "lambda1_dim2": 0.0000,
            "lambda2_dim2": 0.00005,
            "flood_penalty": 0.39
        }

        epochs = 2500


        features = data.get_feature_names(fn_dataset, BMI=False, sex=True, parents_diagnostics=True)
        # print(features)
        
        dataset = data.DiabDataset(fn_dataset, features, random_age=False, soft_label=True, soft_label_alpha=0.02)
        len_trainset = int(0.9*len(dataset))
        trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

        fn_out = f"diabnet/models/model-13-sp-soft-label-randomized-positives-2500-dropout0-bn-decay-flood-hyperopt-{i:03}.pth"
        print("model saved to:", fn_out)
        train(params, trainset, valset, epochs, fn_out, device='cuda')
        # break to train only one model


if __name__ == "__main__":

    net('./datasets/visits_sp_unique_train_randomized_0_positivo_1000_random_0.csv')

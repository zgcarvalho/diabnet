from diabnet import data
from diabnet.train import train
from torch.utils.data import random_split

# DATASET = './datasets/visits_sp_unique_train_positivo_1000_negative_0.csv'

def net(fn_dataset):
    # neural network to SELECTED features with fixed dropout=0.5 during hyperopt
    for i in range(100):
        print(f"Model {i:03}")
        params = {
            "l1_neurons": 256,
            "l2_neurons": 0,
            "l3_neurons": 0,
            "dp0": 0,
            "dp1": 0.5,
            "dp2": 0,
            "dp3": 0,
            "lr": 0.0007,
            "wd": 0.000001,
            "lambda1_dim1":0.0003,
            "lambda2_dim1":0.05,
            "lambda1_dim2":0.0008,
            "lambda2_dim2":0.05
            # to explore new weights
            # "lambda1_dim1":0.0002,
            # "lambda2_dim1":0.03,False
            # "lambda1_dim2":0.0005,
            # "lambda2_dim2":0.03
        }

        epochs = 1000


        features = data.get_feature_names(fn_dataset, BMI=False, sex=True, parents_diagnostics=True)
        # print(features)
        
        dataset = data.DiabDataset(fn_dataset, features, random_age=False, soft_label=True)
        len_trainset = int(0.9*len(dataset))
        trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

        fn_out = f"diabnet/models/model-sp-soft-label-positives-1000-decay-{i:03}.pth'"
        print("model saved to:", fn_out)
        train(params, trainset, valset, epochs, fn_out, device='cuda')
        # break to train only one model


if __name__ == "__main__":

    net('./datasets/visits_sp_unique_train_positivo_1000_negative_0.csv')
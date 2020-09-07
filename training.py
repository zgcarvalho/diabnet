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
            "dropout": 0.50,
            "lr": 0.03,
            "beta1": 0.9,
            "beta2": 0.99,
            "eps": 1e-5,
            "wd": 1e-2,
            "lambda1_dim1": 0.0, #5e-07, 
            "lambda2_dim1": 5e-06,
            "lambda1_dim2": 0.0, #5e-08,
            "lambda2_dim2": 5e-08,
            "flood_penalty": 0.0,
            "soft_label_alpha": 0.1,
            "batch_size": 256  
        }
        

        epochs = 1500


        features = data.get_feature_names(fn_dataset, BMI=False, sex=True, parents_diagnostics=True)
        # print(features)
        
        dataset = data.DiabDataset(fn_dataset, features, soft_label=True, soft_label_alpha=params["soft_label_alpha"])
        len_trainset = int(0.9*len(dataset))
        trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

        fn_out = f"diabnet/models/model-3-soft-label-age-positives-1000-dropout0-bn-decay-flood-hyperopt-batch256-adamw-lc1-20200907-{i:03}.pth"
        print("model saved to:", fn_out)
        train(params, trainset, valset, epochs, fn_out, device='cuda', is_trial=False)
        # end_time = default_timer()
        # break
        # break to train only one model
    # print("Elapsed time = ", end_time - start_time)


if __name__ == "__main__":

    net('./datasets/visits_sp_unique_train_positivo_1000_random_0.csv')
    # net('./datasets/visits_sp_unique_train_positivo_0_random_1000.csv')
    # net('./datasets/visits_sp_unique_train_positivo_0_negatives_1000.csv')
    # net('./datasets/visits_sp_unique_train_randomized_1_positivo_1000_random_0.csv')

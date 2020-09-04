import pandas as pd
from sys import argv

def filter(path, save_to):
    df = pd.read_csv(path).set_index("Unnamed: 0")
    df = df[(df.AGE >= 60)&(df.T2D == 0)]
    df.to_csv(save_to)
    # print(df)

if __name__ == "__main__":
    # filter(
    #   "/home/jgcarvalho/projects/diabnet/datasets/visits_sp_unique_train_positivo_1000_random_0.csv", 
    #   "/home/jgcarvalho/projects/diabnet/datasets/visits_sp_unique_train_positivo_1000_random_0_negatives_older60.csv")
    filter(argv[1], argv[2])
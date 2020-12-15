import pandas as pd
import numpy as np
import torch 
import os
# from diabnet.model import load
from diabnet.ensemble import Ensemble
from diabnet.data import encode_features
# from model import load

class Predictor(object):
    """
    Predictor for type 2 diabetes.
    """
    def __init__(self, ensemble, feature_names, negatives_csv=None):
        """
        Construct a predictor for type 2 diabetes .

        Parameters:
        ensemble(diabnet.Ensemble)
        feature_names(list(strings)): list with feature names
        negatives_csv(string): path to csv file with negatives (pacients without diabetes), ie "diabnet/data/negatives_older60.csv"  
        """
        # TODO assert len feature_names is equal to model input size
        self.ensemble = ensemble
        self.feat_names = feature_names
        self.negative_data = get_negative_data(negatives_csv, self.feat_names) if negatives_csv != None else None

    def patient(self, feat, age=-1, bmi=-1, samples_per_model=1):
        """
        Parameters:
        feat: patient features snps (0,1,2), age and bmi.
        age: if age==-1 feat[age] will be used. else age==n n will be used
        bmi: if bmi==-1 feat[bmi] will be used if it exists. else bmi==n n will be used (this has effect only in NN that use bmi as feature)
        """
        if bmi != -1 and 'BMI' not in self.feat_names:
            print("Warning: BMI is not a feature for this network. Therefore, no impact at predicted values.")
        # features = encode_features(feat, self.feat_names, age, bmi)
        # features = encode_features(self.feat_names, feat)
        # features = torch.unsqueeze(torch.tensor(features, dtype=torch.float), dim=0)
        
        # pass a copy of feature to NOT change original features
        features = self._encode_features(np.copy(feat), age=age, bmi=bmi)
        return self._sampler(features, samples_per_model)

    def negatives(self, age=-1, bmi=-1, samples_per_model=1):
        if bmi != -1 and 'BMI' not in self.feat_names:
            print("Warning: BMI is not a feature for this network. Therefore, no impact at predicted values.")
        if self.negative_data is not None:
            probs = []
            for n in self.negative_data:
                # f = encode_features(n, self.feat_names, age, bmi)
                f = self._encode_features(n, age=age, bmi=bmi)
                # f = encode_features(self.feat_names, n)
                # f = torch.tensor(f, dtype=torch.float)
                p = self._sampler(f, samples_per_model)
                probs.append(p)
            return np.concatenate(probs)
        else:
            print("No information about negatives")
            return None 

    def patient_life(self, feat, age_range=range(20,90,5), bmi=-1, samples_per_model=1):
        probs_by_age = []
        for age in age_range:
            probs_by_age.append(self.patient(feat, age, bmi, samples_per_model))
        return probs_by_age, age_range

    def negatives_life(self, age_range=range(20,90,5), bmi=-1, samples_per_model=1):
        if self.negative_data is not None:   
            probs_by_age = []
            for age in age_range:
                probs_by_age.append(self.negatives(age, bmi, samples_per_model))
            return probs_by_age, age_range
        else:
            print("No information about negatives")
            return None

        
    def _sampler(self, x, samples_per_model):
        return self.ensemble.apply(x, samples_per_model, with_correction=True)

    def _encode_features(self, feat, age=-1, bmi=-1):
        if age >= 0:
            age_idx = self.feat_names.index('AGE')
            feat[age_idx] = age
        if bmi >= 0:
            bmi_idx = self.feat_names.index('BMI')
            feat[bmi_idx] = bmi
        features = encode_features(self.feat_names, feat)
        features = torch.unsqueeze(torch.tensor(features, dtype=torch.float), dim=0)
        return features

def get_negative_data(fn, columns):
    # script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    # abs_file_path = os.path.join(script_dir, fn)
    print("NEGATIVE FILE", fn)
    df = pd.read_csv(fn)
    # feat = df.values[:,2:-1]
    feat = df[columns].values
    return feat

if __name__ == "__main__":
    pass

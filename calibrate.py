from diabnet.data import CalibrationDataset
from glob import glob
from diabnet.model import load, Model
from diabnet import data
from diabnet.calibration import CalibrationModel, ece_mce
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from diabnet.optim import RAdam


def load_ensemble_models(prefix: str) -> Dict[str, Model]:
    ensemble_models = {fn:load(fn) for fn in glob(f'{prefix}-???.pth')}
    return ensemble_models

def calibrate(model: Model, dataset) -> CalibrationModel:
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    m = CalibrationModel(model)
    
    loss_func = BCEWithLogitsLoss()

    # optimization
    optimizer = RAdam(m.parameters(), lr=0.01)
    
    for e in range(200):
        for i, sample in enumerate(loader):
            x, y_true = sample
            
            if e == 0:
                y_pred = model(x)
                ece_orig, mce_orig = ece_mce(y_pred, y_true)
                           
            y_calibrated = m(x)
            ece, mce = ece_mce(y_calibrated, y_true)
            print(f'ECE {ece} MCE {mce} - Uncalibrated ECE {ece_orig} MCE {mce_orig}')
            
            loss = loss_func(y_calibrated, y_true)
            
            optimizer.zero_grad()
            # loss_reg.backward()
            loss.backward()
            optimizer.step()
            
    return m
            
def calibrate_ensemble_models(prefix: str, dataset):
    ensemble_models = load_ensemble_models(prefix)
    for k, v in ensemble_models.items():      
        print(f'Calibrating... {k}')
        fn = f'{prefix}-calibrated-{k[-7:]}'
        model_calibrated = calibrate(v, dataset)
        torch.save(model_calibrated, fn)
        print(f'Saved to {fn}')
        print("done")
        
        
        
if __name__ == "__main__":
    fn_dataset = './datasets/visits_sp_unique_train_positivo_1000_random_0.csv'
    features = data.get_feature_names(fn_dataset, BMI=False, sex=True, parents_diagnostics=True)
    dataset = CalibrationDataset(fn_dataset, features, balance=True, device='cpu')
    
    calibrate_ensemble_models('./diabnet/models/model-13-sp-soft-label-positives-2500-dropout0-bn-decay-flood-hyperopt', dataset)
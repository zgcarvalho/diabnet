from diabnet import data
from diabnet.model import Model
from diabnet.optim import RAdam
from diabnet.calibration import ece_mce
from torch.utils.data import DataLoader, random_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, SGD, AdamW
import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
import datetime
# RADAM
import math
# import torch
# from torch.optim.optimizer import Optimizer, required
from typing import Dict, Any
import pytorch_warmup as warmup
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# def l1_l2_regularization(lc_params, lambda1_dim1, lambda2_dim1, lambda1_dim2, lambda2_dim2):
#     l1_regularization_dim1 = lambda2_dim1*torch.sum(torch.norm(lc_params,1, dim=1)) 
#     l2_regularization_dim1 = (1.0-lambda2_dim1)/2.0*torch.sum(torch.norm(lc_params,2, dim=1))
#     l1_regularization_dim2 = lambda2_dim2*torch.sum(torch.norm(lc_params,1, dim=2))
#     l2_regularization_dim2 = (1.0-lambda2_dim2)/2.0*torch.sum(torch.norm(lc_params,2, dim=2))
    
#     dim1_loss = lambda1_dim1 * (l1_regularization_dim1 + l2_regularization_dim1)
#     dim2_loss = lambda1_dim2 * (l1_regularization_dim2 + l2_regularization_dim2)
    
#     return dim1_loss + dim2_loss

def l1_l2_regularization(lc_params, lambda1_dim1, lambda2_dim1, lambda1_dim2, lambda2_dim2):
    l1_regularization_dim1 = lambda2_dim1*torch.sum(torch.norm(lc_params,1, dim=1)) 
    l2_regularization_dim1 = (1.0-lambda2_dim1)/2.0*torch.sum(torch.norm(lc_params,2, dim=1))
    l1_regularization_dim2 = lambda2_dim2*torch.sum(torch.norm(lc_params,1, dim=2))
    l2_regularization_dim2 = (1.0-lambda2_dim2)/2.0*torch.sum(torch.norm(lc_params,2, dim=2))
    
    dim1_loss = lambda1_dim1 * (l1_regularization_dim1 + l2_regularization_dim1)
    dim2_loss = lambda1_dim2 * (l1_regularization_dim2 + l2_regularization_dim2)
    
    return dim1_loss + dim2_loss
        

def train(params: Dict[str,Any], training_set, validation_set, epochs, fn_to_save_model="", is_trial=False, device='cuda'):
    device=torch.device(device)

    trainloader = DataLoader(training_set, batch_size=params["batch_size"], shuffle=True)
    
    valloader = DataLoader(validation_set, batch_size=len(validation_set), shuffle=False)

    # print("Number of features", training_set.dataset.n_feat)
    model = Model(training_set.dataset.n_feat, params["hidden_neurons"], params["dropout"])
    model.to(device)

    loss_func = BCEWithLogitsLoss()
    loss_func.to(device)

    # optimization
    # optimizer = RAdam(model.parameters(), lr=params["lr"], betas=(params["beta1"], params["beta2"]), eps=params["eps"])
    optimizer = AdamW(model.parameters(), lr=params["lr"], betas=(params["beta1"], params["beta2"]), eps=params["eps"])
    # optimizer = SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4)
    # scheduler = StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=-1)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    

    # lambda to L1 regularization at LC layer
    lambda1_dim1 = params["lambda1_dim1"]
    lambda2_dim1 = params["lambda2_dim1"]
    lambda1_dim2 = params["lambda1_dim2"]
    lambda2_dim2 = params["lambda2_dim2"]

    flood_penalty = params["flood_penalty"]

    for e in range(epochs):
        model.train()
        # trainloader.dataset.random_age = True

        training_loss, training_loss_reg, n_batchs = 0.0, 0.0, 0
        for i, sample in enumerate(trainloader):
            x, y_true = sample
            y_pred = model(x.to(device))
            loss = loss_func(y_pred, y_true.to(device))
            
            # l1 and l2 regularization at LC layer
            loss_reg = loss + l1_l2_regularization(model.lc.weight, lambda1_dim1, lambda2_dim1, lambda1_dim2, lambda2_dim2)
            
            # flood regularization
            # https://arxiv.org/pdf/2002.08709.pdf
            flood = (loss_reg - flood_penalty).abs() + flood_penalty

            optimizer.zero_grad()
            # loss_reg.backward()
            flood.backward()
            optimizer.step()

            training_loss += loss.item()
            training_loss_reg += loss_reg.item()
            n_batchs += 1
            
               
        scheduler.step()        

        training_loss /= n_batchs
        training_loss_reg /= n_batchs

        # don't print epoch loss during hyperparameter optimizations
        if not is_trial:
            print("T epoch {}, loss {}, loss_with_regularization {}".format(e, training_loss, training_loss_reg))


        model.eval() 
            
        for i, sample in enumerate(valloader):
            x, y_true = sample
            y_pred = model(x.to(device))

            loss = loss_func(y_pred, y_true.to(device))
            
            # ece and mce are calibration metrics
            ece, mce = ece_mce(y_pred, y_true)
            # ece, mce = ece.item(), mce.item()

            t = y_true.cpu().detach().numpy()
            p = torch.sigmoid(y_pred).cpu().detach().numpy()
            t_b = t > 0.5
            p_b = p > 0.5
            cm = confusion_matrix(t_b, p_b)
            acc = accuracy_score(t_b, p_b)
            bacc = balanced_accuracy_score(t_b, p_b)
            auroc = roc_auc_score(t_b, p)
            avg_prec = average_precision_score(t_b, p)
            

        if not is_trial:
            print(f"V epoch {e}, loss {loss.item()}, acc {acc:.3}, bacc {bacc:.3}, ece {ece.item():.3}, mce {mce.item():.3}, auc {auroc}, avg_prec {avg_prec}")
            print("line is true, column is pred")
            print(cm)

    if is_trial:
        print("T epoch {}, loss {}, loss_with_regularization {}".format(e, training_loss, training_loss_reg))
        print(f"V epoch {e}, loss {loss.item()}, acc {acc:.3}, bacc {bacc:.3}, ece {ece.item():.3}, mce {mce.item():.3}, auc {auroc}, avg_prec {avg_prec}")
        print(cm)
        print(params)
    

    if fn_to_save_model != "":
        print("Saving model at {}".format(fn_to_save_model))
        torch.save(model, fn_to_save_model)
        with open(fn_to_save_model+'.txt', 'w') as f:
            f.write(str(datetime.datetime.now()))
            f.write(f"\nModel name: {fn_to_save_model}\n")
            f.write(f"\nAccuracy: {acc}\n")
            f.write(f"\nBalanced Accuracy: {bacc}\n")
            f.write(f"\nECE: {ece.item()}\n")
            f.write(f"\nMCE: {mce.item()}\n")
            f.write(f"\nAUC-ROC: {auroc}\n")
            f.write(f"\nAVG-PREC: {avg_prec}\n")
            f.write(f"\nConfusion matrix:\n{cm}\n")
            f.write(f"\nT Loss: {training_loss}\n")
            f.write(f"\nT Loss(reg): {training_loss_reg}\n")
            f.write(f"\nV Loss: {loss.item()}\n\n")
            for k in params:
                f.write("{} = {}\n".format(k,params[k]))
            f.close()
            
    # return validation_loss
    return training_loss, loss.item(), acc, bacc, ece.item(), mce.item(), auroc, avg_prec





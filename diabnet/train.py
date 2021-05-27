import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    fbeta_score,
)
from typing import Dict, Any, Optional
import datetime
from diabnet.model import Model
from diabnet.metrics import ece_mce
from diabnet.data import DiabDataset

__all__ = ["train"]


def _l1_l2_regularization(
    lc_params: torch.nn.parameter.Parameter,
    lambda1_dim1: float,
    lambda2_dim1: float,
    lambda1_dim2: float,
    lambda2_dim2: float,
) -> torch.Tensor:
    # TODO: Document variables
    """Applies L1 (Lasso) and L2 (Ridge) regularization to
    Locally Connected layer of DiabNet. This combined
    regularization avoid overfitting and reduce the
    contributin of less important features to DiabNet.

    Parameters
    ----------
    lc_params : torch.nn.parameter.Parameter
        Parameters of Locally Connected layer of DiabNet.
    lambda1_dim1 : float
        [description]
    lambda2_dim1 : float
        [description]
    lambda1_dim2 : float
        [description]
    lambda2_dim2 : float
        [description]

    Returns
    -------
    torch.Tensor
        L1 and L2 regularized loss.
    """
    l1_regularization_dim1 = lambda2_dim1 * torch.sum(torch.norm(lc_params, 1, dim=1))
    l2_regularization_dim1 = (
        (1.0 - lambda2_dim1) / 2.0 * torch.sum(torch.norm(lc_params, 2, dim=1))
    )
    l1_regularization_dim2 = lambda2_dim2 * torch.sum(torch.norm(lc_params, 1, dim=2))
    l2_regularization_dim2 = (
        (1.0 - lambda2_dim2) / 2.0 * torch.sum(torch.norm(lc_params, 2, dim=2))
    )

    dim1_loss = lambda1_dim1 * (l1_regularization_dim1 + l2_regularization_dim1)
    dim2_loss = lambda1_dim2 * (l1_regularization_dim2 + l2_regularization_dim2)

    return dim1_loss + dim2_loss


def train(
    params: Dict[str, Dict[str, Any]],
    training_set: DiabDataset,
    validation_set: DiabDataset,
    epochs: int,
    prefix: Optional[str] = None,
    logfile: Optional[str] = None,
    is_trial: bool = False,
    device: str = "cuda",
):
    # TODO: Document function + typing return
    """[summary]

    Parameters
    ----------
    params : Dict[str, Dict[str, Any]]
        [description]
    training_set : DiabDataset
        [description]
    validation_set : DiabDataset
        [description]
    epochs : int
        [description]
    prefix : str, optional
        [description], by default None
    logfile : str, optional
        [description], by default None
    is_trial : bool, optional
        [description], by default False
    device : str, optional
        [description], by default "cuda"

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        `optimizer` must be `adamw`.
    """
    # Define the device on which a torch.Tensor will be allocated.
    device = torch.device(device)

    # Create a training set
    trainloader = DataLoader(
        training_set, batch_size=params["batch-size"], shuffle=True
    )

    # Create a validation set
    valloader = DataLoader(
        validation_set, batch_size=len(validation_set), shuffle=False
    )

    # Use soft labels
    use_correction = True

    # Get age index on dataset
    age_idx = training_set.dataset.feat_names.index("AGE")

    # Define DiabNet model
    model = Model(
        training_set.dataset.n_feat,
        params["hidden-neurons"],
        params["dropout"],
        params["lc-layer"],
        use_correction,
        params["soft-label-baseline"],
        params["soft-label-topline"],
        params["soft-label-baseline-slope"],
        age_idx,
    )
    model.to(device)

    # Define loss function
    loss_func = BCEWithLogitsLoss()
    loss_func.to(device)

    # Define optimizer
    if params["optimizer"] == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=params["lr"],
            betas=(params["beta1"], params["beta2"]),
            eps=params["eps"],
            weight_decay=params["wd"],
        )
    else:
        raise ValueError("`optimizer` must be `adamw`.")

    # Define scheduler
    scheduler = StepLR(
        optimizer,
        step_size=params["sched-steps"],
        gamma=params["sched-gamma"],
        last_epoch=-1,
    )

    # lambda to L1 regularization at LC layer
    lambda1_dim1 = params["lambda1-dim1"]
    lambda2_dim1 = params["lambda2-dim1"]
    lambda1_dim2 = params["lambda1-dim2"]
    lambda2_dim2 = params["lambda2-dim2"]

    # Flood regularization (flood penalty)
    # Reference: https://arxiv.org/pdf/2002.08709.pdf
    flood_penalty = params["flood-penalty"]

    # Iterate through epochs
    for e in range(epochs):
        # Activate to training mode
        model.train()

        # Initialize training loss, regularized training loss and number of batches
        training_loss, training_loss_reg, n_batchs = 0.0, 0.0, 0

        # Iterate through training set
        for i, sample in enumerate(trainloader):
            # Get input and true label
            x, y_true = sample

            # Predict label
            y_pred = model(x.to(device))

            # Calculate loss
            loss = loss_func(y_pred, y_true.to(device))

            # l1 and l2 regularization at LC layer
            loss_reg = loss + _l1_l2_regularization(
                model.lc.weight, lambda1_dim1, lambda2_dim1, lambda1_dim2, lambda2_dim2
            )

            # Flood regularization
            # Reference: https://arxiv.org/pdf/2002.08709.pdf
            flood = (loss_reg - flood_penalty).abs() + flood_penalty

            # Sets the gradients of all optimized torch.Tensor s to zero.
            optimizer.zero_grad()

            # Backpropogates the error
            flood.backward()

            # Performs a single optimization step
            optimizer.step()

            # Accumulates loss
            training_loss += loss.item()

            # Accumulates regularized loss
            training_loss_reg += loss_reg.item()

            # Increment number of batches
            n_batchs += 1

        # Updates learning rate of scheduler
        scheduler.step()

        # Calculate (regularized) training loss
        training_loss /= n_batchs
        training_loss_reg /= n_batchs

        # Ignore epoch loss when optimizing hyperparameters
        if not is_trial:
            status = f"T epoch {e}, loss {training_loss}, loss_with_regularization {training_loss_reg}"
            if logfile is None:
                print(status)
            else:
                logfile.write(status + "\n")

        # Activate evaluation mode
        model.eval()

        # Iterate through validation set
        for i, sample in enumerate(valloader):
            # Get input and true label
            x, y_true = sample

            # Predict label
            y_pred = model(x.to(device))

            # Calculate function
            loss = loss_func(y_pred, y_true.to(device))

            # Binarize predictions
            # NOTE: The true labels (y_true) are soft labels. Hence,
            #  convert them to 0 or 1.
            y_ = (y_true > 0.5).type(torch.float)

            # Get ages
            ages = x[:, 0:1, age_idx]

            # Calculate binarized predictions
            # NOTE: The pred labels are probabilities in the interval [0, 1].
            #  Hence, convert them to 0 or 1.
            p = model.sigmoid(y_pred, ages, with_correction=True)

            # Calculate expected calibration error (ECE) and maximum
            #  calibration error (MCE)
            ece, mce = ece_mce(p, y_)

            # Detach true and predicted labels to cpu
            t = y_true.cpu().detach().numpy()
            p = p.cpu().detach().numpy()

            # Calculate metrics
            t_b = t > 0.5
            p_b = p > 0.5
            cm = confusion_matrix(t_b, p_b)
            acc = accuracy_score(t_b, p_b)
            bacc = balanced_accuracy_score(t_b, p_b)
            fscore = fbeta_score(t_b, p_b, beta=1.0)
            auroc = roc_auc_score(t_b, p)
            avg_prec = average_precision_score(t_b, p)

        # Ignore epoch loss when optimizing hyperparameters
        if not is_trial:
            status_0 = f"V epoch {e}, loss {loss.item()}, acc {acc:.3}, bacc {bacc:.3}, ece {ece.item():.3}, mce {mce.item():.3}, auc {auroc}, avg_prec {avg_prec}, fscore {fscore}"
            status_1 = f"line is true, column is pred\n{cm}"
            if logfile is None:
                print(status_0)
                print(status_1)
            else:
                logfile.write(status_0 + "\n")
                logfile.write(status_1 + "\n")

    if prefix is not None:
        # Save trained model
        torch.save(model, f"{prefix}.pth")

        # Save trained model metrics and parameters
        with open(f"{prefix}.txt", "w") as f:
            f.write(str(datetime.datetime.now()))
            f.write(f"\nModel name: {prefix}.pth\n")
            f.write(f"\nAccuracy: {acc}\n")
            f.write(f"\nBalanced Accuracy: {bacc}\n")
            f.write(f"\nF-score: {fscore}\n")
            f.write(f"\nECE: {ece.item()}\n")
            f.write(f"\nMCE: {mce.item()}\n")
            f.write(f"\nAUC-ROC: {auroc}\n")
            f.write(f"\nAVG-PREC: {avg_prec}\n")
            f.write(f"\nConfusion matrix:\n{cm}\n")
            f.write(f"\nT Loss: {training_loss}\n")
            f.write(f"\nT Loss(reg): {training_loss_reg}\n")
            f.write(f"\nV Loss: {loss.item()}\n\n")
            for k in params:
                f.write("{} = {}\n".format(k, params[k]))
            f.close()

    return (
        training_loss,
        loss.item(),
        acc,
        bacc,
        ece.item(),
        mce.item(),
        auroc,
        avg_prec,
        fscore,
    )

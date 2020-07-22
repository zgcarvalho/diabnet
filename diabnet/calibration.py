import torch

def ece_mce(logits, targets, bins=10):
    preds = torch.sigmoid(logits)
    lower_bound = torch.arange(0.0, 1.0, 1.0/bins)
    upper_bound = lower_bound + 1.0/bins
    
    ece = torch.zeros(1, device=preds.device)
    mce = torch.zeros(1, device=preds.device)
    
    for i in range(bins):
        mask = (preds > lower_bound[i]) * (preds <= upper_bound[i])
        if torch.any(mask):
            # print(lower_bound[i], upper_bound[i], preds[mask], torch.mean(targets[mask]))
            delta = torch.abs(torch.mean(preds[mask]) - torch.mean(targets[mask]))
            ece += delta * torch.mean(mask.float())
            mce = torch.max(mce, delta)
            # print(ece, mce)
            
    return ece, mce
    
    
    
    
if __name__ == "__main__":
    p = torch.rand(1000)
    # t = torch.rand(1000)
    t = p + torch.randn(1000)
    ece_mce(p, t, bins=10)
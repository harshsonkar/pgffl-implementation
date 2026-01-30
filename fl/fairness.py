import torch

def gini_coefficient(values):
    """  
    values: list or 1D tensor of client accuracies
    """
    values = torch.tensor(values, dtype=torch.float32)

    if torch.sum(values) == 0:
        return 0.0
    
    diff_sum = torch.sum(torch.abs(values.unsqueeze(0) - values.unsqueeze(1)))
    gini = diff_sum / (2* len(values) * torch.sum(values))

    return gini.item()
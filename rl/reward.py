import torch 

def compute_reward(avg_acc, gini):
    return -avg_acc * torch.log(torch.tensor(gini + 1e-8))
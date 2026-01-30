import torch

def flatten_models(client_models):
    return torch.cat([
        torch.cat([p.flatten() for p in model.parameters()])
        for model in client_models
    ])
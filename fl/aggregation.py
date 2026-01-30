import torch

def weighted_aggregate(client_models, weights):
    global_state = {}

    for key in client_models[0].state_dict().keys():
        global_state[key] = sum(
            [weights[i] * client_models[i].state_dict()[key]
            for i in range(len(client_models))]
        )
        
    return global_state
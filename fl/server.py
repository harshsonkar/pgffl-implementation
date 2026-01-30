import random
import torch
import copy

class Server:
    def __init__(self, global_model, clients, client_data_sizes, device):
        self.global_model = global_model
        self.clients = clients
        self.client_data_sizes = client_data_sizes
        self.device = device

    def select_clients(self, C):
        """
        Randomly select a fraction C of clients
        """
        K = max(1, int(C * len(self.clients)))
        return random.sample(self.clients, K)
    
    def aggregate_fedavg(self, client_models, selected_clients):
        """  
        FedAvg aggregation
        """
        client_weights = []

        for local_model in client_models:
            client_weights.append(local_model.state_dict())

        total_data = sum(
            self.client_data_sizes[c.client_id] for c in selected_clients
        )

        new_state_dict = copy.deepcopy(client_weights[0])

        for key in new_state_dict.keys():
            new_state_dict[key] = torch.zeros_like(new_state_dict[key])

            for weights, client in zip(client_weights, selected_clients):
                pk = self.client_data_sizes[client.client_id] / total_data
                new_state_dict[key] += pk*weights[key]

        self.global_model.load_state_dict(new_state_dict)

    def run_round(self, C, epochs, lr):
        """
        Run ONE federated learning round
        """
        selected_clients = self.select_clients(C)
        
        client_accuracies = []
        client_models = []
        
        for client in selected_clients:
            local_model, acc = client.local_train(
                self.global_model,
                epochs,
                lr
            )
            client_accuracies.append(acc)
            client_models.append(local_model)

        return client_accuracies, client_models, selected_clients

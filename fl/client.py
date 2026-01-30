import copy
import torch
from train.trainer import train_one_epoch, validate

class Client:
    def __init__(self, client_id, train_loader, val_loader, device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
    
    def local_train(self, global_model, epochs, lr):
        """
        Perform local training starting from global model
        """

        # 1. Copy global model -> local model
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)

        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

        # 2. local training
        for _ in range(epochs):
            train_one_epoch(
                local_model,
                self.train_loader,
                optimizer,
                self.device
            )

        # 3. Validation accuracy
        acc = validate(local_model, self.val_loader, self.device)

        # 4. Return updated weights + accuracy
        return local_model, acc

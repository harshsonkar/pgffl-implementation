import torch
import torch.nn.functional as F

def train_one_epoch(model, dataloader, optimizer, device):
    """
    Train model for ONE epoch on local client data
    """

    model.train() # enable training mode

    total_loss=0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()               # clear old gradients
        outputs = model(x)                  # forward pass
        loss = F.cross_entropy(outputs, y)  # compute loss
        loss.backward()                     # backpropagation
        optimizer.step()                    # update parameters

        total_loss+=loss.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    """
    Compute accuracy on validation dataset
    """

    model.eval() # evaluation mode
    correct=0
    total=0

    with torch.no_grad(): # no gradients needed
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            _, predicted = torch.max(outputs, dim=1)

            total += y.size(0)
            correct += (predicted == y).sum().item()
        
    accuracy = correct / total
    return accuracy

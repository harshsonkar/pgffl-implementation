import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=128, num_classes=10):
        super().__init__()

        self.fc1=nn.Linear(input_dim, hidden_dim)
        self.fc2=nn.Linear(hidden_dim, hidden_dim)
        self.fc3=nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [batch_size, 1, 28, 28]
        x=x.view(x.size(0), -1) # flatten

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x) # logits (NO softmax)

        return x
    

# if __name__ == "__main__":
#     model = MLP()
#     dummy = torch.randn(8, 1, 28, 28)
#     out = model(dummy)
#     print(out)
#     print(out.shape)  # should be [8, 10]

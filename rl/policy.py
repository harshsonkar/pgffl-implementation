import torch
import torch.nn as nn
import torch.distributions as D

class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clients):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_clients)
        )

    def forward(self, x):
        """  
        x: concatenated client model parameters 
        return: Gaussian means for each client
        """

        return self.net(x)
    
def sample_weights(policy, state):
    means = policy(state)
    dist = D.Normal(means, torch.ones_like(means))
    actions = dist.sample()
    log_prob = dist.log_prob(actions).sum()

    weights = torch.softmax(actions, dim=0)
    return weights, log_prob

def update_policy(optimizer, log_prob, reward):
    loss = -(log_prob * reward)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
import torch
import random, numpy as np

from data.dataset import generate_client_classes, get_dataloaders, split_iid, split_by_classes
from models.mlp import MLP
from train.trainer import validate

from fl.client import Client
from fl.server import Server
from fl.fairness import gini_coefficient
from fl.aggregation import weighted_aggregate

from rl.policy import PolicyNet, sample_weights, update_policy
from rl.state import flatten_models
from rl.reward import compute_reward

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    # 1. Configurations
    ROUNDS = 40
    NUM_CLIENTS = 5
    CLIENT_FRACTION = 0.6
    LOCAL_EPOCHS = 1
    LR = 0.01
    BATCH_SIZE = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    # 3. create clients (IID split)
    clients = []
    client_data_sizes = [0] * NUM_CLIENTS

    train_datasets = train_loader.dataset
    val_datasets = val_loader.dataset

    # # WHEN: IID SPLIT
    # train_splits = split_iid(train_datasets, NUM_CLIENTS)
    # val_splits = split_iid(val_datasets, NUM_CLIENTS)

    # WHEN: NONIID SPLIT
    targets = train_datasets.dataset.targets[train_datasets.indices]
    num_classes = len(torch.unique(targets))
    client_classes = generate_client_classes(NUM_CLIENTS, num_classes, 2)
    train_splits = split_by_classes(train_datasets, client_classes)
    val_splits   = split_by_classes(val_datasets, client_classes)

    for i in range(NUM_CLIENTS):
        train_dl = torch.utils.data.DataLoader(
            train_splits[i],
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        val_dl = torch.utils.data.DataLoader(
            val_splits[i],
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        client = Client(i, train_dl, val_dl, device)
        clients.append(client)
        client_data_sizes[i] = len(train_splits[i])

    # 4. Initialize global model & Server
    global_model = MLP().to(device)
    server = Server(global_model, clients, client_data_sizes, device)

    # WHEN: PGFFL
    # initalize RL POLICY
    K = max(1, int(CLIENT_FRACTION * NUM_CLIENTS))
    state_dim = flatten_models([global_model]).numel() * K
    policy_net = PolicyNet(
        input_dim=state_dim,
        hidden_dim=128,
        num_clients=NUM_CLIENTS
    ).to(device)
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)

    # 5. Federated Training
    for t in range(ROUNDS):
        client_accuracies, client_models, selected_clients = server.run_round(
            CLIENT_FRACTION,
            LOCAL_EPOCHS,
            LR,
        )

        avg_acc = sum(client_accuracies) / len(client_accuracies)
        gini = gini_coefficient(client_accuracies)
        
        # # WHEN: FedAvg
        # server.aggregate_fedavg(client_models, selected_clients)
        
        # WHEN: PgFFL
        state = flatten_models(client_models).to(device)
        weights, log_prob = sample_weights(policy_net, state)
        new_global_weights = weighted_aggregate(client_models, weights)
        global_model.load_state_dict(new_global_weights)
        reward = compute_reward(avg_acc, gini)
        update_policy(policy_optimizer, log_prob, reward)

        global_test_acc = validate(global_model, test_loader, device)
        print(
            f"Round {t+1}: "
            f"Avg client accuracy = {avg_acc:.4f}, "
            f"Gini = {gini:.4f}, "
            f"Global test acc = {global_test_acc:.4f}"
        )

if __name__ == "__main__":
    main()

import random
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# returns train_loader and val_loader from FashionMNIST
def get_dataloaders(
        data_dir="./data",
        batch_size=64,
        val_split=0.1,
        num_workers=2
):
    
    # 1. Transform: image -> tensor + normalization
    transform = transforms.Compose([
        transforms.ToTensor(),               # (H, W) -> (1, H, W), values [0,1]
        transforms.Normalize((0.5,), (0.5,)) # normalize grayscale
    ])

    # 2. Download dataset
    full_train_dataset=datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset=datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # 3. Train / Validation split
    total_size=len(full_train_dataset)
    val_size=int(val_split*total_size)
    train_size=total_size-val_size

    train_dataset, val_dataset=random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    # 4. DataLoaders
    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader=DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader=DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

def split_iid(dataset, num_clients):
    dataset_split = torch.utils.data.random_split(
        dataset,
        [len(dataset)//num_clients] * num_clients
    )
    return dataset_split

def generate_client_classes(num_clients, num_classes, classes_per_client):
    client_classes = []

    for i in range(num_clients):
        chosen = random.sample(range(num_classes), classes_per_client)
        client_classes.append(chosen)

    # ensure full coverage
    for cls in range(num_classes):
        client_classes[cls % num_clients].append(cls)

    # remove duplicates
    client_classes = [list(set(c)) for c in client_classes]
    return client_classes

def split_by_classes(dataset, client_classes):
    targets = dataset.dataset.targets[dataset.indices]
    client_indices = [[] for _ in client_classes]

    for idx, label in enumerate(targets):
        for i, classes in enumerate(client_classes):
            if label.item() in classes:
                client_indices[i].append(idx)

    return [torch.utils.data.Subset(dataset, idxs) for idxs in client_indices]


import torch
from torch.utils.data import DataLoader, random_split

from data_loader import train_dataset


NUM_CLIENTS = 3
BATCH_SIZE = 32

# Divide dataset among clients
client_size = len(train_dataset) // NUM_CLIENTS

client_datasets = random_split(
    train_dataset,
    [
        client_size,
        client_size,
        len(train_dataset) - (2 * client_size)
    ],
    generator=torch.Generator().manual_seed(42)
)

client_loaders = []

for client_id, dataset in enumerate(client_datasets):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    client_loaders.append(loader)

    print(
        f"Client {client_id + 1}: "
        f"{len(dataset)} images"
    )


print("Total clients:", len(client_loaders))
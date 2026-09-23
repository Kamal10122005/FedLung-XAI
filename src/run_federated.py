
import flwr as fl

from fl_client import LungClient
from fl_server import strategy


def client_fn(cid: str):
    client_id = int(cid)

    client = LungClient(client_id)

    return client.to_client()


if __name__ == "__main__":

    print("Starting Federated Learning...")
    print("Number of clients: 3")
    print("Federated rounds: 3")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(
            num_rounds=3
        ),
        strategy=strategy
    )
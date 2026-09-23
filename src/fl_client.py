
import torch
import flwr as fl

from model import LungCNN
from federated_data import client_loaders
from data_loader import train_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MU = 0.01


class LungClient(fl.client.NumPyClient):

    def __init__(self, client_id):

        self.client_id = client_id
        self.model = LungCNN().to(DEVICE)

        self.train_loader = client_loaders[client_id]

        client_indices = self.train_loader.dataset.indices

        client_labels = [
            train_dataset.targets[index]
            for index in client_indices
        ]

        class_counts = torch.bincount(
            torch.tensor(client_labels),
            minlength=2
        ).float()

        class_counts = torch.clamp(class_counts, min=1)

        class_weights = (
            len(client_labels) / (2 * class_counts)
        ).to(DEVICE)

        self.criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        print(
            f"Client {client_id + 1} class counts: "
            f"{class_counts.tolist()}"
        )

        print(
            f"Client {client_id + 1} class weights: "
            f"{class_weights.tolist()}"
        )

    def get_parameters(self, config):

        return [
            value.cpu().numpy()
            for value in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters):

        state_dict = self.model.state_dict()

        new_state_dict = {
            key: torch.tensor(value)
            for key, value in zip(
                state_dict.keys(),
                parameters
            )
        }

        self.model.load_state_dict(
            new_state_dict,
            strict=True
        )

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        global_parameters = [
            torch.tensor(value, device=DEVICE)
            for value in parameters
        ]

        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            classification_loss = self.criterion(
                outputs,
                labels
            )

            proximal_loss = 0.0

            for local_parameter, global_parameter in zip(
                self.model.parameters(),
                global_parameters
            ):

                proximal_loss += torch.sum(
                    (local_parameter - global_parameter) ** 2
                )

            loss = (
                classification_loss
                + (MU / 2) * proximal_loss
            )

            loss.backward()

            self.optimizer.step()

            total_loss += classification_loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        average_loss = total_loss / len(self.train_loader)

        accuracy = correct / total

        print(
            f"Client {self.client_id + 1} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )

        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {
                "train_loss": average_loss,
                "train_accuracy": accuracy
            }
        )

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        return 0.0, len(self.train_loader.dataset), {}


def client_fn(context):

    client_id = int(
        context.node_config["partition-id"]
    )

    return LungClient(client_id).to_client()


if __name__ == "__main__":

    client = LungClient(0)

    print("FedProx client created successfully!")

    print("Client ID:", client.client_id + 1)

    print(
        "Training images:",
        len(client.train_loader.dataset)
    )
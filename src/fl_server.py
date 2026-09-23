
import torch
import flwr as fl

from model import LungCNN
from data_loader import val_loader


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def get_parameters():
    model = LungCNN()

    return [
        value.cpu().numpy()
        for value in model.state_dict().values()
    ]


def evaluate_model(server_round, parameters, config):

    model = LungCNN().to(DEVICE)

    state_dict = model.state_dict()

    new_state_dict = {
        key: torch.tensor(value)
        for key, value in zip(
            state_dict.keys(),
            parameters
        )
    }

    model.load_state_dict(new_state_dict)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    # Save the latest global model
    torch.save(
        model.state_dict(),
        "fedavg_model.pth"
    )

    print(
        f"Round {server_round} | "
        f"Global Validation Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Global model saved after round "
        f"{server_round}"
    )

    return 1 - accuracy, {
        "val_accuracy": accuracy
    }


def aggregate_fit_metrics(metrics):

    total_examples = sum(
        num_examples * metric["train_accuracy"]
        for num_examples, metric in metrics
    )

    total_samples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    return {
        "train_accuracy": total_examples / total_samples
    }


strategy = fl.server.strategy.FedAvg(

    fraction_fit=1.0,

    fraction_evaluate=0.0,

    min_fit_clients=3,

    min_available_clients=3,

    initial_parameters=fl.common.ndarrays_to_parameters(
        get_parameters()
    ),

    evaluate_fn=evaluate_model,

    fit_metrics_aggregation_fn=aggregate_fit_metrics
)


if __name__ == "__main__":

    print(
        "FedAvg server with validation "
        "evaluation is ready!"
    )
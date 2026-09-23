
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MultiDiseaseMobileNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if pretrained else None
        )

        self.model = mobilenet_v3_small(weights=weights)

        # Freeze pretrained feature extractor initially
        for parameter in self.model.features.parameters():
            parameter.requires_grad = False

        # Replace the original classifier
        input_features = self.model.classifier[0].in_features

        self.model.classifier = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = MultiDiseaseMobileNet(num_classes=4)

    sample = torch.randn(1, 3, 128, 128)
    output = model(sample)

    print("Model: MobileNetV3-Small")
    print("Output shape:", output.shape)
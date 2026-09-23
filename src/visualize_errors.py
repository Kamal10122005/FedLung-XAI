import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("results/multidisease_test_predictions.csv")

errors = df[
    (df["actual"] == "NORMAL") &
    (df["predicted"] == "PNEUMONIA")
].head(12)

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for i, (_, row) in enumerate(errors.iterrows()):
    image = Image.open(row["image_path"]).convert("RGB")

    axes[i].imshow(image, cmap="gray")
    axes[i].set_title(
        f"Actual: {row['actual']}\nPredicted: {row['predicted']}"
    )
    axes[i].axis("off")

plt.suptitle("NORMAL Images Misclassified as PNEUMONIA")
plt.tight_layout()

plt.savefig(
    "results/normal_pneumonia_errors.png",
    dpi=300
)

plt.show()

print("Error visualization saved!")
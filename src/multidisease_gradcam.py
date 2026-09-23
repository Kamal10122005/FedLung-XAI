
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("results/multidisease_test_predictions.csv")

classes = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

cm = confusion_matrix(
    df["actual"],
    df["predicted"],
    labels=classes
)

print("Confusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot(xticks_rotation=45)
plt.title("Multi-Disease Test Confusion Matrix")
plt.tight_layout()

plt.savefig("results/multidisease_confusion_matrix.png", dpi=300)
plt.show()

print("\nConfusion matrix saved!")
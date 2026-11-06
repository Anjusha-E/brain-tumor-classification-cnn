
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset_builder import build_dataset
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from src.config import CLASS_TO_LABEL


# Create folder to save evaluation results
evaluation_dir = Path("artifacts/evaluation")
evaluation_dir.mkdir(parents=True, exist_ok=True)


# Convert label numbers into class names
LABEL_TO_CLASS = {
    value: key
    for key, value in CLASS_TO_LABEL.items()
}

class_names = [
    LABEL_TO_CLASS[i]
    for i in range(len(LABEL_TO_CLASS))
]


# Load the dataset
x_train, y_train, x_test, y_test, train_paths, test_paths = build_dataset()


# Load the trained model
model = load_model("models/brain_tumor_cnn.keras")


# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# Make predictions
y_prob = model.predict(x_test)
y_pred = y_prob.argmax(axis=1)


# Find indexes of incorrectly predicted images
wrong_indices = np.where(y_pred != y_test)[0]

print("Total misclassified images:", len(wrong_indices))


# Display details of the first 10 incorrect predictions
for i in wrong_indices[:10]:
    print(
        f"Index: {i}, "
        f"Actual: {y_test[i]}, "
        f"Predicted: {y_pred[i]}"
    )


# Visualize the first 10 misclassified images
for i in wrong_indices[:10]:
    plt.figure(figsize=(5, 4))

    plt.imshow(x_test[i])

    actual = LABEL_TO_CLASS[int(y_test[i])]
    predicted = LABEL_TO_CLASS[int(y_pred[i])]

    plt.title(f"Actual: {actual} | Predicted: {predicted}")
    plt.axis("off")
    plt.show()
    plt.close()


# Count common misclassification pairs
wrong_pairs = Counter()

for i in wrong_indices:
    actual = LABEL_TO_CLASS[int(y_test[i])]
    predicted = LABEL_TO_CLASS[int(y_pred[i])]

    wrong_pairs[(actual, predicted)] += 1


print("\nMost Common Misclassifications:")

for (actual, predicted), count in wrong_pairs.most_common():
    print(f"{actual} -> {predicted}: {count}")


# Generate confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(class_names)))
)

print("\nConfusion Matrix:")
print(cm)


# Plot confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")
plt.tight_layout()


# Save confusion matrix
plt.savefig(
    evaluation_dir / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# Generate classification report
report = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names
)

print("\nClassification Report:")
print(report)


# Save classification report
with open(
    evaluation_dir / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# Save evaluation summary
with open(
    evaluation_dir / "evaluation_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(f"Test Loss: {test_loss}\n")
    file.write(f"Test Accuracy: {test_accuracy}\n")
    file.write(f"Total Test Images: {len(y_test)}\n")
    file.write(f"Correct Predictions: {len(y_test) - len(wrong_indices)}\n")
    file.write(f"Misclassified Images: {len(wrong_indices)}\n")


print("\nEvaluation results saved successfully!")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(y_test, predictions, classes, save_path=None):
    labels = list(range(len(classes)))
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=labels, target_names=classes, zero_division=0
    ))
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    if save_path:
        # For very large class counts, labels may be crowded; this plot is still saved.
        fig_size = max(7, min(18, len(classes) * 0.35))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticklabels(classes, fontsize=6)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")
        if len(classes) <= 25:
            threshold = matrix.max() / 2 if matrix.size else 0
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                            fontsize=6,
                            color="white" if matrix[i, j] > threshold else "black")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print("Saved confusion matrix:", save_path)
    return accuracy


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print("Saved training history:", save_path)

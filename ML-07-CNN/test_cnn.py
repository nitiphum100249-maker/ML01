import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def test_cnn(n_samples=4):
    model_path = os.path.join(OUTPUT_DIR, "cnn_model.keras")
    test_path = os.path.join(OUTPUT_DIR, "X_test.npy")
    labels_path = os.path.join(OUTPUT_DIR, "y_test.npy")
    classes_path = os.path.join(OUTPUT_DIR, "classes.json")
    for path in (model_path, test_path, labels_path, classes_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required output missing: {path}. Run python main.py first.")
    model = keras.models.load_model(model_path)
    X_test = np.load(test_path)
    y_test = np.load(labels_path)
    with open(classes_path, encoding="utf-8") as f:
        classes = json.load(f)

    n_samples = min(n_samples, len(X_test))
    rng = np.random.default_rng()
    indices = rng.choice(len(X_test), n_samples, replace=False)
    X_sample, y_sample = X_test[indices], y_test[indices]
    probabilities = model.predict(X_sample, verbose=0)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    cols = min(2, n_samples)
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue
        correct = predictions[i] == y_sample[i]
        ax.imshow(X_sample[i])
        ax.axis("off")
        ax.set_title(
            f"Predicted: {classes[predictions[i]]} ({confidence[i]*100:.1f}%)\n"
            f"Actual: {classes[y_sample[i]]} | {'Correct' if correct else 'Incorrect'}"
        )
        print(f"{i+1}. predicted={classes[predictions[i]]}, "
              f"actual={classes[y_sample[i]]}, confidence={confidence[i]*100:.1f}%")
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "prediction_sample.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print("Saved:", output_path)


if __name__ == "__main__":
    test_cnn()

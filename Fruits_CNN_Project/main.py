import json
import os
import numpy as np
from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from cnn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Default: extract the Kaggle archive under dataset/. The loader finds the class folders recursively.
# You can override this with FRUIT_DATA_PATH.
DATA_PATH = os.environ.get("FRUIT_DATA_PATH", os.path.join(BASE_DIR, "dataset"))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
MAX_PER_CLASS = None      # Use all 291 images in Fruits Plant Leaf; set an integer to cap each class.
TEST_SIZE = 0.15
VAL_SIZE = 0.15
EPOCHS = 20
BATCH_SIZE = 32


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("CNN PROJECT: FRUITS PLANT LEAF CLASSIFICATION")
    print("=" * 60)

    print("\n[1/6] Loading fruit-plant leaf images...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)
    print(f"Total images: {len(images)}")
    print(f"Number of classes: {len(classes)}")

    with open(os.path.join(OUTPUT_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)

    print("\n[2/6] Preparing image arrays...")
    X = to_features(images)
    y = labels
    del images

    print("\n[3/6] Splitting data into train / validation / test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )
    del X, y
    for name, array in (
        ("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
        ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)
    ):
        np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), array)
    print(f"Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

    print("\n[4/6] Training CNN...")
    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE
    )

    print("\n[5/6] Evaluating on the test set...")
    predictions = predict_model(model, X_test)
    evaluate_model(
        y_test, predictions, classes,
        os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )
    plot_history(history, os.path.join(OUTPUT_DIR, "training_history.png"))

    print("\n[6/6] Finished. Model and evaluation outputs are in the outputs/ folder.")
    print("Run python test_cnn.py to predict a few test images.")


if __name__ == "__main__":
    main()

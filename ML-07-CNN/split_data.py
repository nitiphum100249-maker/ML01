from sklearn.model_selection import train_test_split
import numpy as np


def split_dataset(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """Stratified split: 70% train, 15% validation, 15% test by default."""
    y = np.asarray(y)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio,
        random_state=random_state, stratify=y_trainval
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

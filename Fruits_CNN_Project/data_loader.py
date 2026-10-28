import os
import cv2
import numpy as np
from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _image_files(folder):
    """List image files directly inside a folder (not recursively)."""
    try:
        return sorted(
            name for name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, name))
            and name.lower().endswith(VALID_EXT)
        )
    except OSError:
        return []


def _count_images_recursive(folder):
    count = 0
    for current, _, files in os.walk(folder):
        count += sum(name.lower().endswith(VALID_EXT) for name in files)
    return count


def resolve_image_root(data_path):
    """Find a folder whose immediate child folders are image classes.

    Supports Kaggle extraction layouts such as:
      dataset/Fruits_Plant_Leaf/Apple/*.jpg
      dataset/Apple/*.jpg
      dataset/Training/Apple/*.jpg
    """
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset folder not found: {os.path.abspath(data_path)}\n"
            "Set FRUIT_DATA_PATH to the extracted Fruits_Plant_Leaf folder, "
            "or to the parent folder containing it."
        )

    preferred_names = ("Fruits_Plant_Leaf", "Training", "training", "train", "images", "Images")
    candidates = [data_path]
    for current, dirs, _ in os.walk(data_path):
        dirs[:] = [d for d in dirs if d not in (".git", "__MACOSX")]
        for dirname in dirs:
            candidates.append(os.path.join(current, dirname))

    def score(folder):
        try:
            children = [os.path.join(folder, d) for d in os.listdir(folder)
                        if os.path.isdir(os.path.join(folder, d))]
        except OSError:
            return (-1, -1, -1)
        class_dirs = [child for child in children if _count_images_recursive(child) > 0]
        # Prefer directories that have many image-bearing child folders, then image count.
        image_count = sum(_count_images_recursive(child) for child in class_dirs)
        name_bonus = 1 if os.path.basename(folder) in preferred_names else 0
        return (len(class_dirs), name_bonus, image_count)

    best = max(candidates, key=score)
    if score(best)[0] >= 2:
        return best

    # Fallback for a single class root with images directly inside it (not expected here).
    if _image_files(data_path):
        raise ValueError(
            f"Images were found directly in {data_path}, but class labels could not be inferred. "
            "Please arrange images into one subfolder per class."
        )
    raise ValueError(
        f"Could not find class folders under {os.path.abspath(data_path)}. "
        "Expected one folder per class (e.g. Apple/*.jpg, Banana/*.jpg)."
    )


def load_data(data_path, img_size=100, max_per_class=None):
    root = resolve_image_root(data_path)
    classes = sorted(
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
        and _count_images_recursive(os.path.join(root, name)) > 0
    )
    if len(classes) < 2:
        raise ValueError(f"Need at least 2 image classes; found {len(classes)} in {root}.")

    print("Image root:", root)
    print("Detected classes:", classes)
    images, labels = [], []
    for label, class_name in enumerate(classes):
        folder = os.path.join(root, class_name)
        paths = []
        for current, _, filenames in os.walk(folder):
            for filename in filenames:
                if filename.lower().endswith(VALID_EXT):
                    paths.append(os.path.join(current, filename))
        paths.sort()
        if max_per_class is not None:
            paths = paths[:max_per_class]
        loaded, skipped = 0, 0
        for path in paths:
            image = preprocess_image(cv2.imread(path), img_size)
            if image is None:
                skipped += 1
                continue
            images.append(image)
            labels.append(label)
            loaded += 1
        print(f"{class_name}: loaded {loaded}, skipped {skipped}")
    if not images:
        raise ValueError("No readable images found. Check the dataset path and image files.")
    return np.stack(images), np.asarray(labels, dtype=np.int64), classes

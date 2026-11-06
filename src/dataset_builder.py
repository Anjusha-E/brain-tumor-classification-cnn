import os
import numpy as np

from src.config import TRAIN_PATH, TEST_PATH, CLASS_TO_LABEL
from src.image_preprocessor import preprocess_image


def load_dataset(folder_path):
    images = []
    labels = []
    image_paths = []

    for class_name in CLASS_TO_LABEL:
        class_path = os.path.join(folder_path, class_name)

        for image_name in sorted(os.listdir(class_path)):
            image_path = os.path.join(class_path, image_name)

            image = preprocess_image(image_path)

            images.append(image)
            labels.append(CLASS_TO_LABEL[class_name])
            image_paths.append(image_path)

    images = np.array(images)
    labels = np.array(labels)
    image_paths = np.array(image_paths)

    # Shuffle images, labels, and paths together
    indices = np.random.default_rng(42).permutation(len(images))

    images = images[indices]
    labels = labels[indices]
    image_paths = image_paths[indices]

    return images, labels, image_paths


    

def build_dataset():
    x_train, y_train, train_paths = load_dataset(TRAIN_PATH)
    x_test, y_test, test_paths = load_dataset(TEST_PATH)

    return (
        x_train,
        y_train,
        x_test,
        y_test,
        train_paths,
        test_paths
    )

# Dataset Paths

TRAIN_PATH = "data/brain_tumor_dataset/Training"
TEST_PATH = "data/brain_tumor_dataset/Testing"


# Image Configuration

IMAGE_SIZE = (224, 224)


# Class Names

CLASS_TO_LABEL = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}

# Number of Classes

NUM_CLASSES = len(CLASS_TO_LABEL)
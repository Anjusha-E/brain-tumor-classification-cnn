
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset Paths
TRAIN_PATH = "data/brain_tumor_dataset/Training"
TEST_PATH = "data/brain_tumor_dataset/Testing"

# Model Path
MODEL_PATH = BASE_DIR / "models" / "brain_tumor_cnn.keras"

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

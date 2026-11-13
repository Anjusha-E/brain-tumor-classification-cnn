
from pathlib import Path
from functools import lru_cache
import os
import requests

from tensorflow.keras.models import load_model


# Get the main project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the model location
MODEL_PATH = PROJECT_ROOT / "models" / "brain_tumor_cnn.keras"

# GitHub Release download URL
MODEL_URL = (
    "https://github.com/Anjusha-E/brain-tumor-classification-cnn/"
    "releases/download/v1.0.0/brain_tumor_cnn.keras"
)


def download_model():

    # Create the models directory if it does not exist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Temporary file for downloading
    temp_path = MODEL_PATH.with_name(MODEL_PATH.name + ".download")

    print("Model not found locally.")
    print("Downloading model from GitHub Releases...")

    try:
        # Download the model in smaller chunks
        with requests.get(
            MODEL_URL,
            stream=True,
            timeout=(15, 120)
        ) as response:

            response.raise_for_status()

            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        file.write(chunk)

        # Check that the downloaded file is not empty
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError("Downloaded model file is empty.")

        # Move the completed download to the final location
        os.replace(temp_path, MODEL_PATH)

        print("Model downloaded successfully.")

    except Exception as error:

        # Remove incomplete download
        if temp_path.exists():
            temp_path.unlink()

        raise RuntimeError(
            f"Model download failed: {error}"
        ) from error


@lru_cache(maxsize=1)
def get_model():

    # Download the model if it is missing
    if not MODEL_PATH.exists():
        download_model()

    print("Loading CNN model...")

    # Load and return the trained model
    model = load_model(str(MODEL_PATH))

    print("CNN model loaded successfully.")

    return model

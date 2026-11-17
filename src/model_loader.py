
from pathlib import Path
from functools import lru_cache
import hashlib
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

# Expected SHA-256 checksum
EXPECTED_SHA256 = (
    "1fd54c4caf4373ac8c6f873512850b73e3c193dcf85dd169f337d219f2d6f00e"
)


def calculate_sha256(file_path):

    # Create a SHA-256 hash object
    sha256 = hashlib.sha256()

    # Read the file in smaller chunks
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def verify_model(file_path):

    # Calculate the model's checksum
    actual_hash = calculate_sha256(file_path)

    # Compare with the expected checksum
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(
            "Model integrity verification failed. "
            "The model file may be corrupted or modified."
        )

    print("Model integrity verified successfully.")

    return True


def download_model():

    # Create models directory if it does not exist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Temporary download file
    temp_path = MODEL_PATH.with_name(
        MODEL_PATH.name + ".download"
    )

    print("Model not found locally.")
    print("Downloading model from GitHub Releases...")

    try:
        # Download the model
        with requests.get(
            MODEL_URL,
            stream=True,
            timeout=(15, 120)
        ) as response:

            response.raise_for_status()

            with open(temp_path, "wb") as file:

                for chunk in response.iter_content(
                    chunk_size=8 * 1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        # Verify downloaded file
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError("Downloaded model file is empty.")

        print("Verifying downloaded model...")

        verify_model(temp_path)

        # Move verified model to final location
        os.replace(temp_path, MODEL_PATH)

        print("Model downloaded successfully.")

    except Exception as error:

        # Remove incomplete or invalid download
        if temp_path.exists():
            temp_path.unlink()

        raise RuntimeError(
            f"Model download failed: {error}"
        ) from error


@lru_cache(maxsize=1)
def get_model():

    # Download if the model is missing
    if not MODEL_PATH.exists():
        download_model()

    else:
        print("Checking existing model integrity...")

        try:
            verify_model(MODEL_PATH)

        except RuntimeError:

            print("Existing model is invalid.")
            print("Downloading a verified copy...")

            MODEL_PATH.unlink()
            download_model()

    print("Loading CNN model...")

    # Load the verified model
    model = load_model(str(MODEL_PATH))

    print("CNN model loaded successfully.")

    return model

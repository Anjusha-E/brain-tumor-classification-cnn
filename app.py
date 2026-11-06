
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
from src.image_preprocessor import preprocess_image_array
from tensorflow.keras.models import load_model
from src.config import CLASS_TO_LABEL

# Create the FastAPI application
app = FastAPI()

# Load the trained CNN model
model = load_model("models/brain_tumor_cnn.keras")

# Convert numeric labels into class names
LABEL_TO_CLASS = {
    value: key
    for key, value in CLASS_TO_LABEL.items()
}


# API home endpoint
@app.get("/")
def home():
    return {"message": "Brain Tumor Detection API is running"}


# Predict the category of an uploaded brain MRI image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read the uploaded image bytes
    contents = await file.read()

    # Convert image bytes into a NumPy array
    image_array = np.frombuffer(contents, np.uint8)

    # Decode the image using OpenCV (BGR format)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Reject invalid image files
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # Preprocess the image exactly once
    image = preprocess_image_array(image)

    # Add the batch dimension required by the CNN
    image = np.expand_dims(image, axis=0)

    # Generate class probabilities
    probabilities = model.predict(image, verbose=0)

    # Find the class with the highest probability
    predicted_label = int(np.argmax(probabilities[0]))

    # Convert the predicted label into its class name
    predicted_class = LABEL_TO_CLASS[predicted_label]

    # Calculate the highest predicted probability
    confidence = float(probabilities[0][predicted_label]) * 100

    # Return the prediction as JSON
    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2)
    }

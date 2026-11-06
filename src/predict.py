import sys
from tensorflow.keras.models import load_model
from src.image_preprocessor import preprocess_image
from src.config import CLASS_TO_LABEL


# Load the trained model
model = load_model("models/brain_tumor_cnn.keras")


# Create label-to-class mapping
LABEL_TO_CLASS = {
    value: key
    for key, value in CLASS_TO_LABEL.items()
}


def predict_image(image_path):

    # Preprocess the image
    image = preprocess_image(image_path)

    # Add batch dimension
    image = image.reshape(1, 224, 224, 3)

    # Predict
    probabilities = model.predict(image)

    # Get predicted class
    predicted_label = probabilities.argmax(axis=1)[0]

    # Convert label to class name
    predicted_class = LABEL_TO_CLASS[predicted_label]

    # Get confidence of predicted class
    confidence = probabilities[0][predicted_label] * 100

    print("Predicted Class:", predicted_class)
    print("Confidence:", f"{confidence:.2f}%")

    print("\nClass Probabilities:")

    for label, class_name in LABEL_TO_CLASS.items():
        probability = probabilities[0][label] * 100
        print(f"{class_name}: {probability:.2f}%")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide an image path.")
    else:
        image_path = sys.argv[1]
        predict_image(image_path)
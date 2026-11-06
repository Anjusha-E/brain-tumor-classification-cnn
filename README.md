# Brain MRI Classification System

A deep learning-based image classification application that uses a Convolutional Neural Network (CNN) to classify brain MRI images into four categories: Glioma, Meningioma, No Tumor, and Pituitary.

The application combines TensorFlow and Keras for deep learning, FastAPI for backend image processing and predictions, and Streamlit for an interactive user interface.

**Note:** This project is developed for educational and research purposes only. It is not intended for clinical diagnosis or medical decision-making.

## Features

- Classifies brain MRI images into four predefined categories.
- Uses a CNN model built with TensorFlow and Keras.
- Performs image preprocessing using OpenCV and NumPy.
- Provides an interactive dashboard using Streamlit.
- Uses FastAPI to handle image uploads and predictions.
- Displays the predicted category and confidence score.
- Includes a separate About Project page.

## Classification Categories

The model classifies brain MRI images into the following four categories:

| Category | Description |
|---|---|
| Glioma | A type of tumor originating from glial cells. |
| Meningioma | A tumor that develops in the meninges surrounding the brain and spinal cord. |
| No Tumor | MRI images labelled as having no tumor in the dataset. |
| Pituitary | A tumor involving the pituitary gland. |

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras, CNN |
| Image Processing | OpenCV, NumPy |
| Frontend | Streamlit |
| Backend | FastAPI, Uvicorn |
| Machine Learning Evaluation | Scikit-learn |
| Data Visualization | Matplotlib, Seaborn |
| Development Environment | Visual Studio Code |

## Project Architecture

The application follows a client-server architecture.

### Workflow

1. The user uploads a brain MRI image through the Streamlit interface.
2. Streamlit sends the image to the FastAPI backend.
3. FastAPI receives and preprocesses the image.
4. The preprocessed image is passed to the trained CNN model.
5. The CNN predicts one of the four predefined categories.
6. FastAPI returns the predicted category and confidence score.
7. Streamlit displays the prediction result.

## Dataset Description

The project uses a brain MRI image dataset divided into Training and Testing directories.

### Dataset Structure

```text
data/
└── brain_tumor_dataset/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    │
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

- **Training:** Used to train the CNN model.
- **Testing:** Used to evaluate the trained model on separate images.
- **Four Classes:** Both directories contain images belonging to the four classification categories.

The dataset is stored locally and excluded from the Git repository.

## CNN Model Architecture

A Convolutional Neural Network (CNN) is used to extract image features and classify brain MRI scans.

### Model Configuration

| Parameter | Value |
|---|---|
| Input Image Size | 224 × 224 × 3 |
| Number of Classes | 4 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy |

### Model Layers

| Layer | Configuration |
|---|---|
| Input | 224 × 224 × 3 |
| Conv2D | 32 filters |
| MaxPooling2D | Default pool size |
| Conv2D | 64 filters |
| MaxPooling2D | Default pool size |
| Flatten | Feature vector conversion |
| Dense | 128 neurons |
| Output | 4 neurons with Softmax activation |

The Softmax activation function produces probabilities for the four categories. The category with the highest predicted probability is selected as the output.

## Model Evaluation

The trained CNN model was evaluated using the separate testing dataset.

### Evaluation Results

| Metric | Result |
|---|---:|
| Test Accuracy | 95.12% |
| Test Loss | 0.1954 |
| Total Test Images | 1,311 |
| Correct Predictions | 1,247 |
| Incorrect Predictions | 64 |

The model achieved approximately 95.12% test accuracy, correctly classifying 1,247 out of 1,311 test images.

These results represent performance on the available testing dataset and do not establish clinical reliability.

## Project Structure

```text
BRAIN_TUMOR_DETECTION_CNN/
│
├── app.py
├── streamlit_app.py
│
├── src/
│   ├── config.py
│   ├── image_preprocessor.py
│   ├── dataset_builder.py
│   ├── model_builder.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── brain_tumor_cnn.keras
│
├── data/
│   └── brain_tumor_dataset/
│
├── notebooks/
├── artifacts/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Anjusha-E/brain-tumor-classification-cnn.git
```

### 2. Navigate to the Project Directory

```bash
cd BRAIN_TUMOR_DETECTION_CNN
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Prepare the Model

Obtain the trained model file `brain_tumor_cnn.keras` and place it inside the `models/` directory.

The trained model and dataset are excluded from the Git repository and must be obtained separately.

## Running the Application

The application requires two separate terminals because the frontend and backend run independently.

### 1. Start the FastAPI Backend

Open the first terminal and run:

```bash
uvicorn app:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

### 2. Start the Streamlit Frontend

Open a second terminal, activate the virtual environment, and run:

```bash
streamlit run streamlit_app.py
```

Frontend URL:

```text
http://localhost:8501
```

### 3. Use the Application

1. Open the Streamlit application.
2. Upload a brain MRI image.
3. Submit the image for classification.
4. View the predicted category and confidence score.

## API Endpoint

The FastAPI backend provides an image classification endpoint.

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Accepts an uploaded image and returns the predicted category and confidence score. |

### Example Response

```json
{
    "filename": "brain_mri.jpg",
    "predicted_class": "meningioma",
    "confidence": 99.32
}
```

## Limitations

- The model is trained on a specific brain MRI dataset.
- Predictions may be unreliable for images that differ significantly from the training data.
- The model may assign unrelated images to one of its four predefined categories.
- Prediction confidence does not guarantee correctness.
- This application is not clinically validated and must not be used as a substitute for professional medical evaluation.

## Future Improvements

- Improve model generalization using data augmentation and additional datasets.
- Explore transfer learning with pretrained CNN architectures.
- Implement validation to identify unsupported image inputs.
- Enhance model evaluation and visualization.
- Deploy the application for online access.

## Author

**Anjusha E.**

- GitHub: [Anjusha-E](https://github.com/Anjusha-E)
- LinkedIn: [Anjusha E](https://www.linkedin.com/in/anjusha-e-a73382370/)
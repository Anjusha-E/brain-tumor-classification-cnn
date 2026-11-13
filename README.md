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
- Automatically downloads the trained model from GitHub Releases if it is not available locally.
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
| HTTP Requests | Requests |
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

### Model Loading

The trained CNN model is hosted separately in GitHub Releases because of its large file size.

The project uses a shared model loader (`src/model_loader.py`) to manage model loading:

1. Checks whether the trained model exists in the local `models/` directory.
2. If the model exists, it loads the local file.
3. If the model is missing, it downloads the model automatically from GitHub Releases.
4. Saves the downloaded model in the `models/` directory.
5. Loads the model for prediction.

Users do not need to download the model manually.

**Trained Model:** [Download brain_tumor_cnn.keras](https://github.com/Anjusha-E/brain-tumor-classification-cnn/releases/download/v1.0.0/brain_tumor_cnn.keras)

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

The dataset is stored locally and excluded from the Git repository. Users who want to retrain or reevaluate the model must provide the dataset in the expected directory structure.

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

These results represent performance on the available testing dataset and do not establish clinical reliability. In particular, the model's recall for the glioma class was approximately 88%, meaning some actual glioma images were misclassified during testing.

## Project Structure

```text
brain-tumor-classification-cnn/
│
├── app.py
├── streamlit_app.py
│
├── src/
│   ├── config.py
│   ├── image_preprocessor.py
│   ├── model_loader.py
│   ├── dataset_builder.py
│   ├── model_builder.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── brain_tumor_cnn.keras  # Downloaded automatically if missing
│
├── data/
│   └── brain_tumor_dataset/   # Local dataset, not included in Git
│
├── notebooks/
├── artifacts/
│   └── evaluation/
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
cd brain-tumor-classification-cnn
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

No manual model download is required.

When the application starts, `src/model_loader.py` checks whether the trained model exists locally. If it is missing, the model is automatically downloaded from the GitHub Release and saved in the `models/` directory.

An internet connection is required for the initial download.

## Running the Application

The application requires two separate terminals because the frontend and backend run independently.

### 1. Start the FastAPI Backend

Open the first terminal and run:

```bash
uvicorn app:app --reload
```

Backend URL:

http://127.0.0.1:8000

API Documentation:

http://127.0.0.1:8000/docs

### 2. Start the Streamlit Frontend

Open a second terminal, activate the virtual environment, and run:

```bash
streamlit run streamlit_app.py
```

Frontend URL:

http://localhost:8501

### 3. Use the Application

1. Open the Streamlit application.
2. Upload a brain MRI image.
3. Submit the image for classification.
4. View the predicted category and confidence score.

## API Endpoint

The FastAPI backend provides an image classification endpoint.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Checks whether the API is running. |
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
- The model has not been clinically validated.
- This application must not be used as a substitute for professional medical evaluation.

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
- Project Repository: [Brain MRI Classification System](https://github.com/Anjusha-E/brain-tumor-classification-cnn)

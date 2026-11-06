#import libraries 
import cv2
import numpy as np
from src.config import IMAGE_SIZE

def read_image(image_path):
   """
   Read an image from the specified path.
   Parameters:
        image_path (str): Path to the image.

    Returns:
        numpy.ndarray: Loaded image.
   """ 
   image = cv2.imread(image_path)
   if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
   return image

def convert_to_rgb(image):
    # Convert BGR image to RGB
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def resize_image(image):
    #resize the image to the specified size
    
    return cv2.resize(image, IMAGE_SIZE)

def normalize_image(image):
    #normalize the pixel values to the range [0, 1]
    return image.astype(np.float32) / 255.0

def preprocess_image(image_path):
    """
    Complete preprocessing pipeline.

    Steps:
        1. Read image
        2. Convert BGR to RGB
        3. Resize image
        4. Normalize image

    Returns:
        Processed image
    """
    image = read_image(image_path)
    image = convert_to_rgb(image)
    image = resize_image(image)
    image = normalize_image(image)
    return image

def preprocess_image_array(image):
    image = convert_to_rgb(image)
    image = resize_image(image)
    image = normalize_image(image)

    return image
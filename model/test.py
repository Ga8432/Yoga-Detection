import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('best_model.h5')

# Define the class labels (modify according to your classes)
class_labels = ['downdog', 'goddess', 'plank', 'tree','warrior2']

def preprocess_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))  # Adjust the size based on your model requirements
    image = image / 255.0  # Normalize pixel values

    return np.expand_dims(image, axis=0)

def predict_yoga_posture(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Make predictions
    predictions = model.predict(processed_image)

    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]

    return predicted_class


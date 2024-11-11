import numpy as np
import tensorflow as tf
import cv2

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(file):
    # Read the image file from the uploaded file object
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize the image to the target size (224, 224)
    img = cv2.resize(img, (150, 150))

    # Normalize the image: convert pixel values from [0, 255] to [0, 1]
    img = img.astype('float32') / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

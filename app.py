from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from my_model import load_model, preprocess_image

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # Allow requests from your React app


# Load your model
model = load_model('my_model.h5')
class_names = ['Recyclable', 'Non-Recyclable']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file:
        return jsonify({'error': 'File is empty'}), 400

    # Read and preprocess the image
    img = preprocess_image(file)

    # Use img directly for prediction
    predictions = model.predict(img)

    # Process predictions and convert to native Python types
    predicted_class_index = int(np.argmax(predictions))

    predicted_class_name = class_names[predicted_class_index]

    return jsonify({'predicted_class': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)

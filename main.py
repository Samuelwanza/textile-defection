from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import tempfile
from tensorflow.keras.models import load_model
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
#on loading the app, download the model
model_url = 'https://drive.google.com/uc?id=1QNVnlNmUaUcs8s4XeKNJGjaB5XMLHJ7r'

model_path = 'textile_model.h5'
response = requests.get(model_url)
with open(model_path, 'wb') as f:
    f.write(response.content)
model = load_model(model_path)

# Set up a temporary directory for storing images
temp_image_dir = tempfile.mkdtemp()

@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['image']

        # Save the image to a temporary file
        temp_image_path = os.path.join(temp_image_dir, 'temp_image.png')
        file.save(temp_image_path)

        # Read the image file
        image = Image.open(temp_image_path).convert("L")  # Convert to grayscale
        image = image.resize((224, 224))

        # Convert the image to a numpy array
        input_image = np.array(image) / 255.0
        input_image = np.expand_dims(input_image, axis=-1)  # Add color channel dimension
        input_image = np.repeat(input_image, 3, axis=-1)  # Duplicate the channel to create three identical channels
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(input_image)

        # Get the class with the highest probability
        predicted_class = np.argmax(predictions)

        # Map the class index to your actual class labels
        class_labels = {0: 'good', 1: 'damaged'}  # Adjust based on your classes

        # Get the predicted class label
        predicted_label = class_labels[predicted_class]

        # Return the prediction as JSON
        return jsonify({'prediction': predicted_label})

    finally:
        # Cleanup: Remove the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))

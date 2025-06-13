from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model("model.h5")  # Pre-trained LSTM model

# Define prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Example: expecting input as a JSON with all features in a list
        input_features = [
            data['air_temperature'],
            data['relative_humidity'],
            data['air_pressure'],
            data['wind_speed'],
            data['hour'],
            data['dayofweek'],
            data['month']
        ]

        # Reshape input to match LSTM shape: (1, timesteps, features)
        input_array = np.array(input_features).reshape(1, 1, len(input_features))
        prediction = model.predict(input_array)[0][0]

        return jsonify({'prediction': float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route("/", methods=["GET"])
def home():
    return "Energy Prediction API Running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
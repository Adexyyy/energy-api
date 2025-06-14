from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model without compiling to avoid deserialization issues on Render
model = load_model("model.h5", compile=False)

# Define prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Expecting input as a JSON with a "sequence" key containing a list of dicts
        sequence = data["sequence"]  # List of 24 dicts

        # Extract features in the correct order for each time step
        feature_names = [
            "air_temperature",
            "relative_humidity",
            "air_pressure",
            "wind_speed",
            "hour",
            "dayofweek",
            "month"
        ]
        # Build a list of lists: shape (24, 7)
        input_features = [
            [step[feature] for feature in feature_names]
            for step in sequence
        ]

        # Reshape to (1, 24, 7) for batch size 1
        input_array = np.array(input_features).reshape(1, 24, 7)
        prediction = model.predict(input_array)[0][0]

        return jsonify({'prediction': float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route("/", methods=["GET"])
def home():
    return "Energy Prediction API Running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
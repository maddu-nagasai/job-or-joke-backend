from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:8080"}})  # Allow frontend access

# Load trained model
model = joblib.load("job_detection_model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "Fake Job Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get JSON data from frontend
    job_description = data.get("description", "")

    if not job_description:
        return jsonify({"error": "Job description is required!"}), 400

    # Make prediction
    prediction = model.predict([job_description])[0]
    result = "Fake Job Posting" if prediction == 1 else "Legitimate Job Posting"

    return jsonify({"prediction": result})

# Run the API
if __name__ == "__main__":
    app.run()

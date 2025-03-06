from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

# Ensure models directory exists
if not os.path.exists("models"):
    raise FileNotFoundError("❌ Models directory not found. Run 'tune_model.py' first.")

# Load best model
model_path = "models/random_forest_best_model.pkl"  # Change to best model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file {model_path} not found. Train your model first.")

model = joblib.load(model_path)

# Load the scaler (if used during training)
scaler_path = "models/scaler.pkl"
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# Load feature names from training data
df_train = pd.read_csv("data/cleaned_customer_churn.csv")
expected_features = df_train.drop(columns=['churn']).columns.tolist()

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Customer Churn Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if request contains JSON
        if not request.is_json:
            return jsonify({"error": "Invalid JSON format. Ensure the request contains valid JSON data."}), 400

        # Get JSON data from request
        data = request.get_json()

        # Validate "features" key in JSON
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request JSON."}), 400
        
        # Convert input to DataFrame
        input_data = pd.DataFrame([data["features"]], columns=expected_features)

        # Scale data if necessary
        if scaler:
            input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Probability of churn

        # Return result as JSON
        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": round(probability, 4)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from geopy.distance import geodesic

app = Flask(__name__)

# Load models and scaler
models = {
    "decision_tree": joblib.load("models/decision_tree.pkl"),
    "knn": joblib.load("models/knn.pkl"),
    "linear_regression": joblib.load("models/linear_regression.pkl"),
    "neural_network": joblib.load("models/neural_network.pkl"),
    "random_forest": joblib.load("models/random_forest.pkl")
}
scaler = joblib.load("models/scaler.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from form
        data = request.json
        pickup_lat = float(data["pickup_lat"])
        pickup_lon = float(data["pickup_lon"])
        dropoff_lat = float(data["dropoff_lat"])
        dropoff_lon = float(data["dropoff_lon"])

        # Create input data
        features = pd.DataFrame([{
            "pickup_hour": 8,  # Example static hour, can make this dynamic
            "pickup_dayofweek": 1,  # Example static day of the week
            "pickup_month": 6,  # Example static month
            "dropoff_hour": 9,  # Example static hour
            "dropoff_dayofweek": 1,  # Example static day of the week
            "dropoff_month": 6,  # Example static month
            "pickup_longitude": pickup_lon,
            "pickup_latitude": pickup_lat,
            "dropoff_longitude": dropoff_lon,
            "dropoff_latitude": dropoff_lat,
            "passenger_count": 1
        }])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict using all models
        predictions = {
            model_name: model.predict(features_scaled if model_name != "linear_regression" else features)[0]
            for model_name, model in models.items()
        }

        # Return the predictions
        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

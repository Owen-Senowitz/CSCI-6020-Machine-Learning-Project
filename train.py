import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Start timing the script
start_time = time.time()

# Create a directory for saving models
os.makedirs("models", exist_ok=True)

print("Loading dataset...")
# Load dataset
data = pd.read_csv("data/trip_data.csv")
print("Dataset loaded successfully!")

# Convert pickup and dropoff datetime columns to datetime objects
print("Converting datetime columns...")
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])
print("Datetime columns converted!")

# Extract features
print("Extracting features from datetime columns...")
data['pickup_hour'] = data['pickup_datetime'].dt.hour
data['pickup_dayofweek'] = data['pickup_datetime'].dt.dayofweek
data['pickup_month'] = data['pickup_datetime'].dt.month
data['dropoff_hour'] = data['dropoff_datetime'].dt.hour
data['dropoff_dayofweek'] = data['dropoff_datetime'].dt.dayofweek
data['dropoff_month'] = data['dropoff_datetime'].dt.month
print("Features extracted!")

# Define X and y
print("Preparing feature matrix and target variable...")
X = data[['pickup_hour', 'pickup_dayofweek', 'pickup_month', 
          'dropoff_hour', 'dropoff_dayofweek', 'dropoff_month',
          'pickup_longitude', 'pickup_latitude', 
          'dropoff_longitude', 'dropoff_latitude', 
          'passenger_count']]
y = data['trip_duration']
print(f"Feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features normalized!")

# Train models
models = {
    "decision_tree": DecisionTreeRegressor(),
    "knn": KNeighborsRegressor(n_neighbors=5),
    "linear_regression": LinearRegression(),
    "neural_network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

print("Training models...")
for model_name, model in models.items():
    print(f"Training {model_name}...")
    if model_name == "linear_regression":
        model.fit(X_train, y_train)
    else:
        model.fit(X_train_scaled, y_train)
    print(f"{model_name} trained successfully!")

# Save models
print("Saving models to 'models' directory...")
for model_name, model in models.items():
    joblib.dump(model, f"models/{model_name}.pkl")
    print(f"{model_name} saved to models/{model_name}.pkl")

# Save the scaler
print("Saving scaler...")
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved to models/scaler.pkl")

# Calculate total execution time
end_time = time.time()
total_time = end_time - start_time
minutes, seconds = divmod(total_time, 60)
print(f"Training and saving completed successfully in {int(minutes)} minutes and {int(seconds)} seconds!")

import pandas as pd
from joblib import load

# Load the trained model
model = load("models/random_forest.joblib")

# Load or create new data for prediction
new_data = pd.read_csv("data/raw/new_data.csv")

# Apply the same preprocessing as in training
from data_processing import engineer_features
features = engineer_features(new_data)

# Make predictions
X = features.drop(columns=["CustomerId"], errors="ignore")
predictions = model.predict_proba(X)[:, 1]  # Probability of high risk

# Output results
output = pd.DataFrame({
    "CustomerId": features["CustomerId"],
    "risk_probability": predictions
})
output.to_csv("data/processed/predictions.csv", index=False)


from fastapi import FastAPI
from pydantic import ValidationError
import pandas as pd
import mlflow.sklearn

from src.api.pydantic_models import CustomerData, RiskResponse

app = FastAPI()

# Load the latest registered model from MLflow registry
model_name = "random_forest"
model_uri = f"models:/{model_name}/Production"
model = mlflow.sklearn.load_model(model_uri)


@app.get("/")
def root():
    return {"message": "Credit risk scoring API is up."}


@app.post("/predict", response_model=RiskResponse)
def predict(data: CustomerData):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Perform prediction (risk probability)
        risk_proba = model.predict_proba(input_df)[0][1]

        return RiskResponse(risk_probability=risk_proba)

    except ValidationError as ve:
        return {"error": str(ve)}

import pandas as pd
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Set MLflow tracking URI to local server
import mlflow.sklearn
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Import feature engineering and target labeling functions
from data_processing import engineer_features
from proxy_target_engineering import (
    calculate_rfm,
    segment_customers_rfm,
    add_target_to_dataset
)


def load_data(path="data/raw/data.csv"):
    # Load raw dataset from specified path
    return pd.read_csv(path)


def evaluate_model(y_true, y_pred, y_proba):
    # Compute standard evaluation metrics for classification
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }


def train_models():
    # Step 1: Load raw data
    raw_df = load_data()

    # Step 2: Generate RFM-based proxy targets
    rfm = calculate_rfm(raw_df)
    risk_labels = segment_customers_rfm(rfm)

    # Step 3: Engineer features and attach proxy target labels
    features_df = engineer_features(raw_df)
    final_df = add_target_to_dataset(features_df, risk_labels)

    # Step 4: Prepare feature matrix (X) and labels (y)
    X = final_df.drop(columns=["CustomerId", "is_high_risk"], errors="ignore")
    X = X.select_dtypes(include=["number"])  # Keep only numeric columns
    y = final_df["is_high_risk"]

    # Drop rows with missing values to ensure model compatibility
    X = X.dropna()
    y = y.loc[X.index]  # Align y with filtered X

    # Step 5: Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 6: Define models and their hyperparameter search grids
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier()
    }

    param_grid = {
        "logistic_regression": {"C": [0.1, 1.0, 10.0]},
        "random_forest": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None]
        }
    }

    best_model = None
    best_score = 0
    best_name = ""

    # Step 7: Train and evaluate each model using cross-validation
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(model, param_grid[name], cv=3, scoring="f1")
            search.fit(X_train, y_train)

            # Predict on test set and evaluate
            y_pred = search.predict(X_test)
            y_proba = search.predict_proba(X_test)[:, 1]
            scores = evaluate_model(y_test, y_pred, y_proba)

            # Log hyperparameters and evaluation metrics
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(scores)

            # Log the trained model to MLflow (and register it)
            mlflow.sklearn.log_model(
                sk_model=search.best_estimator_,
                artifact_path=name,
                registered_model_name=name
            )

            # Track the best-performing model based on F1 score
            if scores["f1"] > best_score:
                best_score = scores["f1"]
                best_model = search.best_estimator_
                best_name = name

    # Step 8: Save the best model locally
    print(f" Best model: {best_name} with F1 score: {best_score:.4f}")
    dump(best_model, f"models/{best_name}.joblib")


# Entry point
if __name__ == "__main__":
    train_models()

# Credit Scoring Business Understanding
1. Basel II and the Need for Interpretable Models: The Basel II Accord emphasizes accurate risk measurement and regulatory compliance. This requires credit risk models to be transparent, interpretable, and well-documented, so that both internal stakeholders and regulators can understand how risk is assessed and ensure fair, consistent lending decisions.

2. Proxy Variables for Default and Associated Risks: In many datasets, a direct "default" label is missing. To train models, we create a proxy variable—for example, labeling a customer as defaulted if they miss payments for 90+ days. However, if this proxy doesn’t accurately reflect real-world defaults, it can lead to misclassification, biased decisions, and financial or reputational risk.

3. Trade-offs: Simple vs. Complex Models

Simple models (e.g., Logistic Regression with WoE) are easy to explain and audit, making them ideal for regulated environments.

Complex models (e.g., Gradient Boosting) often deliver higher predictive accuracy but are harder to interpret. The key trade-off is between performance and explainability—in finance, interpretability often takes priority to meet compliance and build trust.

# Credit Risk Scoring Pipeline

This project presents an end-to-end machine learning pipeline for simulating credit risk prediction using transaction-level data.

## Key Features

- **Feature Engineering**  
  Derived temporal features, aggregated customer behavior, and preprocessed numerical/categorical data using a modular scikit-learn pipeline.

- **Proxy Target Generation**  
  Applied RFM-based clustering with KMeans to define high-risk customer segments in the absence of labeled credit default data.

- **Model Training & Evaluation**  
  Trained Logistic Regression and Random Forest classifiers. Random Forest outperformed across accuracy, F1 score, and ROC AUC, and was selected for deployment.

- **Model Deployment**  
  Exposed the trained model as a RESTful API using FastAPI, with prediction capability via `/predict` endpoint. Model is loaded directly from MLflow Model Registry.

- **Containerization & CI/CD**  
  Dockerized the service and integrated automated testing and linting via GitHub Actions to enforce code quality and maintainability.

## Tech Stack

- Python, scikit-learn, pandas
- MLflow for experiment tracking & model registry
- FastAPI & Uvicorn for API serving
- Docker + docker-compose
- GitHub Actions (CI/CD)
- flake8 & pytest for code quality

---

This project demonstrates applied machine learning principles for real-world modeling, reproducibility, and deployment workflows.

## Model Selection and Performance

I trained both a Logistic Regression model and a Random Forest Classifier, comparing them using cross-validation and multiple performance metrics. While Logistic Regression offered slightly higher precision, its F1 score was extremely low-highlighting an imbalance in predictive performance.

**Final Model: Random Forest Classifier**  
It delivered the best balance between precision and recall, with the following results:

- Accuracy: 0.73
- F1 Score: 0.51
- Precision: 0.59
- Recall: 0.45
- ROC AUC: 0.77

The model was registered in MLflow and deployed via a FastAPI endpoint for serving predictions.


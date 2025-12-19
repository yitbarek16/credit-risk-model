# 💳 Credit Risk Probability Modeling with Alternative Data

## 📘 Overview

Traditional credit scoring relies on historical loan repayment data, which excludes many customers with limited or no credit history. To support a **Buy-Now-Pay-Later (BNPL)** service, **Bati Bank** partnered with an eCommerce platform to assess customer creditworthiness using **alternative behavioral data**.

This project builds an **end-to-end credit risk modeling system** that transforms transaction behavior into a **credit risk probability**, a **credit score**, and **loan recommendations**. The solution combines feature engineering, machine learning, and MLOps best practices to deliver a production-ready credit scoring pipeline.

This project was completed as part of the **10 Academy – Artificial Intelligence Mastery Program (Week 5 Challenge)**.


## 🎯 Business Problem

Bati Bank needs a reliable way to decide:

* **Who qualifies** for BNPL credit
* **How risky** each customer is
* **How much credit** to offer and for how long

The challenge is that the dataset does **not contain a direct default label**. The solution must therefore:

* Define a **proxy for credit risk**
* Use behavioral patterns to predict default likelihood
* Produce explainable and regulator-friendly results in line with **Basel II principles**


## 📂 Dataset

The dataset comes from an eCommerce transaction platform and includes **customer-level behavioral data**, such as:

* Transaction history (amounts, frequency, recency)
* Product categories and providers
* Channels used (web, mobile, BNPL)
* Pricing strategy and fraud indicators
* Time-based transaction patterns

This data enables **behavior-based credit modeling** without traditional loan records.


## 🔍 What This Project Does

### 1️⃣ Proxy Default Definition

* Used **RFM (Recency, Frequency, Monetary)** analysis to represent customer risk behavior
* Applied **KMeans clustering** to segment customers
* Labeled high-risk clusters as *bad* and low-risk clusters as *good*

This created a **proxy default variable** suitable for supervised learning.

### 2️⃣ Feature Engineering

* Aggregated transaction data at the customer level
* Engineered behavioral, temporal, and monetary features
* Built a modular **scikit-learn pipeline** for preprocessing and modeling

### 3️⃣ Credit Risk Modeling

* Trained and evaluated:

  * **Logistic Regression** (interpretable baseline)
  * **Random Forest Classifier** (performance-focused)
* Compared models using:

  * Accuracy
  * Precision / Recall
  * F1-score
  * ROC AUC

**Random Forest** achieved the best balance and was selected.

### 4️⃣ Credit Score & Loan Estimation

* Converted predicted **risk probabilities** into:

  * A **credit score**
  * Recommended **loan amount**
  * Suggested **loan duration**

This enables actionable BNPL decisions, not just predictions.

### 5️⃣ Deployment & MLOps

* Tracked experiments and registered models using **MLflow**
* Exposed the trained model via a **FastAPI REST API**
* Dockerized the service for reproducibility
* Integrated **CI/CD** with GitHub Actions
* Implemented logging, testing, and linting for production readiness


## 📊 Model Performance (Final Model)

**Random Forest Classifier**

* Accuracy: **0.73**
* F1 Score: **0.51**
* Precision: **0.59**
* Recall: **0.45**
* ROC AUC: **0.77**

The model delivers a strong balance between risk detection and false positives, suitable for credit decision workflows.


## 🚀 Outcome

The final system provides Bati Bank with:

* A **risk probability score** for each customer
* A derived **credit score**
* Data-driven **loan amount and duration recommendations**
* A fully deployable and auditable ML pipeline


## 🧰 Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **MLflow**
* **FastAPI & Uvicorn**
* **Docker & Docker Compose**
* **GitHub Actions (CI/CD)**
* **Pytest, Flake8**
* **Logging & Model Registry**


## 🛡️ License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and share this project with proper attribution.

---
Let's stay in touch! Feel free to connect with me on LinkedIn:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yitbarektesfaye)

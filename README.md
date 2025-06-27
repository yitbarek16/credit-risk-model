# Credit Scoring Business Understanding
1. Basel II and the Need for Interpretable Models: The Basel II Accord emphasizes accurate risk measurement and regulatory compliance. This requires credit risk models to be transparent, interpretable, and well-documented, so that both internal stakeholders and regulators can understand how risk is assessed and ensure fair, consistent lending decisions.

2. Proxy Variables for Default and Associated Risks: In many datasets, a direct "default" label is missing. To train models, we create a proxy variable—for example, labeling a customer as defaulted if they miss payments for 90+ days. However, if this proxy doesn’t accurately reflect real-world defaults, it can lead to misclassification, biased decisions, and financial or reputational risk.

3. Trade-offs: Simple vs. Complex Models

Simple models (e.g., Logistic Regression with WoE) are easy to explain and audit, making them ideal for regulated environments.

Complex models (e.g., Gradient Boosting) often deliver higher predictive accuracy but are harder to interpret. The key trade-off is between performance and explainability—in finance, interpretability often takes priority to meet compliance and build trust.
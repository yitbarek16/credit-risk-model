import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


# Custom transformer to aggregate transaction statistics per customer
class CustomerAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Aggregate total, average, count, and standard deviation of transaction amounts
        agg = X.groupby("CustomerId")["Amount"].agg(["sum", "mean", "count", "std"]).reset_index()
        agg.columns = ["CustomerId", "total_amount", "avg_amount", "txn_count", "std_amount"]
        return agg


# Custom transformer to extract time-based features from transaction timestamps
class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, time_column="TransactionStartTime"):
        self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.time_column] = pd.to_datetime(X[self.time_column], errors="coerce")
        # Extract hour, day, month, and year from the timestamp
        X["txn_hour"] = X[self.time_column].dt.hour
        X["txn_day"] = X[self.time_column].dt.day
        X["txn_month"] = X[self.time_column].dt.month
        X["txn_year"] = X[self.time_column].dt.year
        return X.drop(columns=[self.time_column])


# Builds preprocessing pipeline including time feature extraction and categorical/numerical transformations
def build_pipeline():
    num_cols = ["Amount", "Value", "PricingStrategy", "FraudResult",
                "txn_hour", "txn_day", "txn_month", "txn_year"]

    cat_cols = ["ProductCategory", "ChannelId", "ProviderId"]

    # Define known categories to ensure consistent encoding
    known_product_categories = [
        "airtime", "data_bundles", "financial_services",
        "movies", "other", "ticket", "transport", "tv", "utility_bill"
    ]
    known_channel_ids = ["ChannelId_1", "ChannelId_2", "ChannelId_3", "ChannelId_5"]
    known_provider_ids = [
        "ProviderId_1", "ProviderId_2", "ProviderId_3",
        "ProviderId_4", "ProviderId_5", "ProviderId_6"
    ]

    # Define categorical pipeline: impute missing values and encode
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, categories=[
            known_product_categories,
            known_channel_ids,
            known_provider_ids
        ]))
    ])

    # Define numerical pipeline: impute and scale
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Combine both into a column transformer
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # Full preprocessing pipeline including time features
    return Pipeline([
        ("time_features", TimeFeaturesExtractor()),
        ("preprocessing", preprocessor)
    ])


# Main feature engineering function
def engineer_features(df):
    # Compute customer-level transaction stats
    customer_stats = CustomerAggregator().fit_transform(df)

    # Fit and apply full transformation pipeline
    pipeline = build_pipeline()
    pipeline.fit(df)
    processed_array = pipeline.transform(df)

    # Retrieve feature names from transformers
    pre = pipeline.named_steps["preprocessing"]
    num_features = pre.named_transformers_["num"].get_feature_names_out()
    cat_features = pre.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out()
    feature_names = list(num_features) + list(cat_features)

    # Handle possible mismatch between array shape and feature names
    print("Number of feature names:", len(feature_names))
    print("Feature names list:", feature_names)
    if processed_array.shape[1] != len(feature_names):
        print("Feature name count mismatch!")
        feature_names = [f"f{i}" for i in range(processed_array.shape[1])]

    # Combine processed features with aggregated stats and customer ID
    processed_df = pd.DataFrame(processed_array, columns=feature_names)
    result = pd.concat(
        [processed_df, customer_stats.drop(columns="CustomerId", errors="ignore")],
        axis=1
    )
    result["CustomerId"] = customer_stats["CustomerId"]

    return result

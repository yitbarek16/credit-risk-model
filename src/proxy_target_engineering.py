import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def calculate_rfm(df, snapshot_date="2019-02-14"):
    # Convert transaction time to datetime with UTC timezone
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], utc=True)

    # Ensure snapshot date is timezone-aware
    snapshot = pd.to_datetime(snapshot_date).tz_localize("UTC") if pd.to_datetime(snapshot_date).tzinfo is None else pd.to_datetime(snapshot_date)

    # Calculate Recency (days since last transaction), Frequency (count), and Monetary (total spend) per customer
    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot - x.max()).days,
        "TransactionId": "count",
        "Amount": "sum"
    }).rename(columns={
        "TransactionStartTime": "Recency",
        "TransactionId": "Frequency",
        "Amount": "Monetary"
    })

    return rfm

def segment_customers_rfm(rfm, random_state=42):
    rfm = rfm.copy()

    # Standardize RFM features for clustering
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # Cluster customers using KMeans (3 segments)
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init="auto")
    rfm["Cluster"] = kmeans.fit_predict(scaled)

    # Identify the high-risk cluster: low frequency, low monetary, high recency
    cluster_means = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    high_risk_cluster = cluster_means.sort_values(
        by=["Frequency", "Monetary", "Recency"],
        ascending=[True, True, False]
    ).index[0]

    # Create binary risk label (1 = high risk, 0 = not high risk)
    rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)

    return rfm.reset_index()[["CustomerId", "is_high_risk"]]

def add_target_to_dataset(features_df, risk_labels_df):
    # Merge risk labels into feature dataset, defaulting to 0 for missing customers
    return features_df.merge(risk_labels_df, on="CustomerId", how="left").fillna({"is_high_risk": 0})

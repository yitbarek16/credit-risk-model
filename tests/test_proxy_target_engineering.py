import pandas as pd
from src.proxy_target_engineering import (
    calculate_rfm,
    segment_customers_rfm,
    add_target_to_dataset
)


def test_proxy_target_pipeline_runs():
    # Step 1: Define a minimal synthetic dataset with 3 transactions
    sample_data = pd.DataFrame([
        {
            "TransactionId": "TransactionId_76871",
            "BatchId": "BatchId_36123",
            "AccountId": "AccountId_3957",
            "SubscriptionId": "SubscriptionId_887",
            "CustomerId": 4406,
            "CurrencyCode": "UGX",
            "CountryCode": "256",
            "ProviderId": "ProviderId_6",
            "ProductId": "ProductId_10",
            "ProductCategory": "airtime",
            "ChannelId": "ChannelId_3",
            "Amount": 1000,
            "Value": 1000,
            "TransactionStartTime": "2018-11-15T02:18:49Z",
            "PricingStrategy": 2,
            "FraudResult": 0
        },
        {
            "TransactionId": "TransactionId_76872",
            "BatchId": "BatchId_36124",
            "AccountId": "AccountId_3958",
            "SubscriptionId": "SubscriptionId_888",
            "CustomerId": 4407,
            "CurrencyCode": "UGX",
            "CountryCode": "256",
            "ProviderId": "ProviderId_5",
            "ProductId": "ProductId_11",
            "ProductCategory": "airtime",
            "ChannelId": "ChannelId_2",
            "Amount": 500,
            "Value": 500,
            "TransactionStartTime": "2019-01-01T14:30:00Z",
            "PricingStrategy": 1,
            "FraudResult": 0
        },
        {
            "TransactionId": "TransactionId_76873",
            "BatchId": "BatchId_36125",
            "AccountId": "AccountId_3959",
            "SubscriptionId": "SubscriptionId_889",
            "CustomerId": 4408,
            "CurrencyCode": "UGX",
            "CountryCode": "256",
            "ProviderId": "ProviderId_7",
            "ProductId": "ProductId_12",
            "ProductCategory": "airtime",
            "ChannelId": "ChannelId_1",
            "Amount": 250,
            "Value": 250,
            "TransactionStartTime": "2019-01-10T09:00:00Z",
            "PricingStrategy": 2,
            "FraudResult": 0
        }
    ])

    # Step 2: Generate Recency-Frequency-Monetary (RFM) metrics
    rfm = calculate_rfm(sample_data)

    # Step 3: Segment customers into risk groups
    labels = segment_customers_rfm(rfm)

    # Step 4: Add target labels to dataset
    merged = add_target_to_dataset(sample_data, labels)

    # Step 5: Assertions
    assert not merged.empty
    assert "is_high_risk" in merged.columns
    assert merged["is_high_risk"].isin([0, 1]).all()

import pandas as pd
from data_processing import engineer_features


def test_engineer_features():
    # Load a sample raw dataset from the specified path
    df = pd.read_csv("data/raw/data.csv")

    # Run the complete feature engineering pipeline
    result = engineer_features(df)

    # Assertions to confirm pipeline integrity
    assert not result.empty  # Ensure the output DataFrame is not empty
    assert "txn_hour" in result.columns  # Time-based feature from timestamp should be present
    assert "total_amount" in result.columns  # Aggregated customer-level feature should exist

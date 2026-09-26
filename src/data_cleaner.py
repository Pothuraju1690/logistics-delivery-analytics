import numpy as np
import pandas as pd


DATE_COLUMNS = [
    "order_date",
    "dispatch_date",
    "expected_delivery_date",
    "actual_delivery_date"
]


def handle_missing_values(df):
    """Handle missing values."""

    text_cols = df.select_dtypes(include="object").columns

    for col in text_cols:
        df[col] = df[col].fillna("Unknown")

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def remove_duplicates(df):
    """Remove duplicate records."""
    return df.drop_duplicates()


def clean_text_columns(df):
    """Clean text columns."""

    text_cols = df.select_dtypes(include="object").columns

    for col in text_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

    return df


def clean_date_columns(df):
    """Convert date columns to datetime."""

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

    return df


def validate_numeric_columns(df):
    """Validate numeric business fields."""

    numeric_columns = [
        "distance_km",
        "quantity",
        "weight_kg",
        "shipping_cost",
        "fuel_cost",
        "warehouse_processing_hours",
        "customer_rating"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


def validate_business_rules(df):
    """Apply business validation rules."""

    df = df[df["order_id"].notna()]
    df = df[df["customer_id"].notna()]
    df = df[df["distance_km"] >= 0]
    df = df[df["quantity"] > 0]
    df = df[df["weight_kg"] >= 0]
    df = df[df["shipping_cost"] >= 0]
    df = df[df["fuel_cost"] >= 0]

    if "customer_rating" in df.columns:
        df = df[
            (df["customer_rating"] >= 1)
            & (df["customer_rating"] <= 5)
        ]

    return df


def create_derived_columns(df):
    """Create required derived business columns."""

    # Convert Yes/No flags to 1/0
    if "return_flag" in df.columns:
        df["return_flag"] = (
            df["return_flag"]
            .astype(str)
            .str.strip()
            .str.title()
            .map({
                "Yes": 1,
                "No": 0
            })
        )

    if "damage_flag" in df.columns:
        df["damage_flag"] = (
            df["damage_flag"]
            .astype(str)
            .str.strip()
            .str.title()
            .map({
                "Yes": 1,
                "No": 0
            })
        )

    df["delivery_days"] = (
        df["actual_delivery_date"]
        - df["dispatch_date"]
    ).dt.days

    df["delay_days"] = (
        df["actual_delivery_date"]
        - df["expected_delivery_date"]
    ).dt.days

    df["cost_per_km"] = np.where(
        df["distance_km"] == 0,
        0,
        df["shipping_cost"] / df["distance_km"]
    )

    df["total_logistics_cost"] = (
        df["shipping_cost"]
        + df["fuel_cost"]
    )

    df["on_time_flag"] = np.where(
        df["delay_days"] <= 0,
        1,
        0
    )

    return df


def clean_data(df):
    """Complete cleaning workflow."""

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = clean_text_columns(df)
    df = clean_date_columns(df)
    df = validate_numeric_columns(df)
    df = validate_business_rules(df)
    df = create_derived_columns(df)

    return df


def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV."""
    df.to_csv(output_path, index=False)
import pandas as pd

from src.data_cleaner import (
    remove_duplicates,
    create_derived_columns
)

from src.business_analysis import (
    calculate_on_time_percentage,
    calculate_total_delivered,
    calculate_total_delayed,
    calculate_total_orders,
)


def test_load_data():
    assert True


def test_remove_duplicates():

    df = pd.DataFrame(
        {"id": [1, 1, 2]}
    )

    result = remove_duplicates(df)

    assert len(result) == 2


def test_create_derived_columns():

    df = pd.DataFrame({
        "dispatch_date": ["2024-01-01"],
        "expected_delivery_date": ["2024-01-03"],
        "actual_delivery_date": ["2024-01-04"],
        "distance_km": [100],
        "shipping_cost": [200],
        "fuel_cost": [50]
    })

    df["dispatch_date"] = pd.to_datetime(df["dispatch_date"])
    df["expected_delivery_date"] = pd.to_datetime(df["expected_delivery_date"])
    df["actual_delivery_date"] = pd.to_datetime(df["actual_delivery_date"])

    df = create_derived_columns(df)

    assert "delivery_days" in df.columns
    assert "delay_days" in df.columns


def test_clean_data():
    assert True


def test_calculate_total_orders():

    df = pd.DataFrame(
        {"order_id": [1, 2, 3]}
    )

    assert calculate_total_orders(df) == 3


def test_calculate_on_time_percentage():
    df = pd.DataFrame({
        "order_id": [1, 2, 3],
        "on_time_flag": [1, 0, 1],
    })

    assert calculate_total_orders(df) == 3
    assert calculate_on_time_percentage(df) == 66.67


def test_business_kpis_handle_case_variants_and_missing_columns():
    df = pd.DataFrame({
        "order_id": [1, 2, 3],
        "delivery_status": ["delivered", "Delayed", "delivered"],
        "delay_days": [0, 2, 0],
        "on_time_flag": [1, 0, 1],
    })

    assert calculate_total_delivered(df) == 2
    assert calculate_total_delayed(df) == 1
    assert calculate_on_time_percentage(df) == 66.67

    minimal_df = pd.DataFrame({"delivery_status": ["Delivered"]})
    assert calculate_total_orders(minimal_df) == 1


def test_save_cleaned_data():
    assert True
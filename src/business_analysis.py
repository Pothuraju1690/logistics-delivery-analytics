import pandas as pd


def calculate_total_orders(df):
    if "order_id" not in df.columns:
        return len(df)
    return df["order_id"].nunique()


def calculate_total_delivered(df):
    if "delivery_status" not in df.columns:
        return 0
    return (
        df["delivery_status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("delivered")
        .sum()
    )


def calculate_total_delayed(df):
    if "delivery_status" in df.columns:
        delayed = (
            df["delivery_status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("delayed")
            .sum()
        )
        if delayed > 0:
            return delayed

    if "delay_days" not in df.columns:
        return 0

    return (pd.to_numeric(df["delay_days"], errors="coerce") > 0).sum()


def calculate_on_time_percentage(df):
    total_orders = calculate_total_orders(df)

    if total_orders == 0:
        return 0

    if "on_time_flag" in df.columns:
        on_time = pd.to_numeric(df["on_time_flag"], errors="coerce").fillna(0)
        return round((on_time.sum() / total_orders) * 100, 2)

    if "delivery_status" in df.columns:
        delivered = calculate_total_delivered(df)
        return round((delivered / total_orders) * 100, 2)

    return 0


def calculate_average_delivery_days(df):
    if "delivery_days" not in df.columns:
        return 0.0
    return round(pd.to_numeric(df["delivery_days"], errors="coerce").mean(), 2)

def calculate_average_delay_days(df):
    delayed = df[df["delay_days"] > 0]

    if delayed.empty:
        return 0

    return round(delayed["delay_days"].mean(), 2)

def calculate_total_shipping_cost(df):
    return df["shipping_cost"].sum()

def calculate_average_shipping_cost(df):
    return df["shipping_cost"].mean()

def calculate_total_fuel_cost(df):
    return df["fuel_cost"].sum()

def calculate_total_logistics_cost(df):
    return df["total_logistics_cost"].sum()

def calculate_average_logistics_cost(df):
    return df["total_logistics_cost"].mean()

def calculate_average_cost_per_km(df):
    return df["cost_per_km"].mean()

def analyze_route_performance(df):
    return df.groupby(
        ["origin_city", "destination_city"]
    ).size().reset_index(name="shipment_count")

def analyze_route_cost(df):
    return df.groupby(
        ["origin_city", "destination_city"]
    )["total_logistics_cost"].mean().reset_index()

def analyze_route_delays(df):
    return df.groupby(
        ["origin_city", "destination_city"]
    )["delay_days"].mean().reset_index()

def analyze_warehouse_performance(df):
    return df.groupby("warehouse").size()

def analyze_warehouse_delays(df):
    return df.groupby("warehouse")["delay_days"].mean()

def analyze_warehouse_processing_time(df):
    return df.groupby("warehouse")[
        "warehouse_processing_hours"
    ].mean()

def analyze_shipping_mode(df):
    return df.groupby("shipping_mode").size()

def analyze_shipping_mode_cost(df):
    return df.groupby("shipping_mode")[
        "total_logistics_cost"
    ].mean()

def analyze_shipping_mode_delays(df):
    return df.groupby("shipping_mode")[
        "delay_days"
    ].mean()

def analyze_delivery_partner(df):
    return df.groupby("delivery_partner").size()

def analyze_partner_cost(df):
    return df.groupby("delivery_partner")[
        "total_logistics_cost"
    ].mean()

def analyze_partner_delays(df):
    return df.groupby("delivery_partner")[
        "delay_days"
    ].mean()

def analyze_partner_damage_rate(df):
    return df.groupby("delivery_partner")[
        "damage_flag"
    ].mean() * 100

def calculate_average_customer_rating(df):
    return df["customer_rating"].mean()

def calculate_return_rate(df):
    return round(
        (df["return_flag"].sum() / len(df)) * 100,
        2
    )


def calculate_damage_rate(df):
    return round(
        (df["damage_flag"].sum() / len(df)) * 100,
        2
    )
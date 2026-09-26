import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")


def plot_delivery_status(df):
    """Plot delivery status distribution."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="delivery_status")
    plt.title("Delivery Status Distribution")
    plt.xlabel("Delivery Status")
    plt.ylabel("Count")
    plt.show()


def plot_monthly_orders(df):
    """Plot monthly order trend."""
    monthly = df.groupby(
        df["order_date"].dt.to_period("M")
    ).size()

    monthly.index = monthly.index.astype(str)

    plt.figure(figsize=(10, 5))
    monthly.plot(marker="o")
    plt.title("Monthly Orders Trend")
    plt.xlabel("Month")
    plt.ylabel("Orders")
    plt.show()


def plot_monthly_delivery_performance(df):
    """Plot monthly on-time deliveries."""

    monthly = (
        df.groupby(df["order_date"].dt.to_period("M"))
        ["on_time_flag"]
        .mean()
        * 100
    )

    monthly.index = monthly.index.astype(str)

    plt.figure(figsize=(10, 5))
    monthly.plot(marker="o", color="green")
    plt.title("Monthly Delivery Performance")
    plt.xlabel("Month")
    plt.ylabel("On-Time %")
    plt.show()


def plot_delay_by_warehouse(df):
    """Plot delay by warehouse."""

    delay = (
        df.groupby("warehouse")["delay_days"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(x=delay.index, y=delay.values)
    plt.title("Average Delay by Warehouse")
    plt.xlabel("Warehouse")
    plt.ylabel("Average Delay Days")
    plt.xticks(rotation=45)
    plt.show()


def plot_delay_by_shipping_mode(df):
    """Plot delay by shipping mode."""

    delay = (
        df.groupby("shipping_mode")["delay_days"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(x=delay.index, y=delay.values)
    plt.title("Average Delay by Shipping Mode")
    plt.xlabel("Shipping Mode")
    plt.ylabel("Delay Days")
    plt.show()


def plot_delay_by_route(df):
    """Plot delay by route."""

    route_delay = (
        df.groupby(
            ["origin_city", "destination_city"]
        )["delay_days"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    route_names = [
        f"{i} → {j}"
        for i, j in route_delay.index
    ]

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=route_delay.values,
        y=route_names
    )
    plt.title("Top 10 Routes by Delay")
    plt.xlabel("Delay Days")
    plt.ylabel("Route")
    plt.show()


def plot_delay_distribution(df):
    """Plot delay distribution."""

    plt.figure(figsize=(8, 5))
    sns.histplot(df["delay_days"], bins=20, kde=True)
    plt.title("Delay Distribution")
    plt.xlabel("Delay Days")
    plt.ylabel("Frequency")
    plt.show()


def plot_cost_by_shipping_mode(df):
    """Plot cost by shipping mode."""

    data = (
        df.groupby("shipping_mode")
        ["total_logistics_cost"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(x=data.index, y=data.values)
    plt.title("Average Cost by Shipping Mode")
    plt.xlabel("Shipping Mode")
    plt.ylabel("Cost")
    plt.show()


def plot_cost_by_route(df):
    """Plot route costs."""

    route_cost = (
        df.groupby(
            ["origin_city", "destination_city"]
        )["total_logistics_cost"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    names = [
        f"{o} → {d}"
        for o, d in route_cost.index
    ]

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=route_cost.values,
        y=names
    )
    plt.title("Top 10 Route Costs")
    plt.xlabel("Average Cost")
    plt.ylabel("Route")
    plt.show()


def plot_cost_trend(df):
    """Plot logistics cost trend."""

    trend = (
        df.groupby(
            df["order_date"].dt.to_period("M")
        )["total_logistics_cost"]
        .sum()
    )

    trend.index = trend.index.astype(str)

    plt.figure(figsize=(10, 5))
    trend.plot(marker="o")
    plt.title("Total Logistics Cost Trend")
    plt.xlabel("Month")
    plt.ylabel("Cost")
    plt.show()


def plot_orders_by_warehouse(df):
    """Plot orders by warehouse."""

    warehouse_orders = (
        df["warehouse"]
        .value_counts()
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=warehouse_orders.index,
        y=warehouse_orders.values
    )
    plt.title("Orders by Warehouse")
    plt.xlabel("Warehouse")
    plt.ylabel("Orders")
    plt.xticks(rotation=45)
    plt.show()


def plot_processing_time_by_warehouse(df):
    """Plot warehouse processing time."""

    processing = (
        df.groupby("warehouse")
        ["warehouse_processing_hours"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=processing.index,
        y=processing.values
    )
    plt.title("Warehouse Processing Time")
    plt.xlabel("Warehouse")
    plt.ylabel("Hours")
    plt.xticks(rotation=45)
    plt.show()


def plot_shipments_by_partner(df):
    """Plot shipments by partner."""

    partner_count = (
        df["delivery_partner"]
        .value_counts()
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=partner_count.index,
        y=partner_count.values
    )
    plt.title("Shipments by Partner")
    plt.xlabel("Delivery Partner")
    plt.ylabel("Shipments")
    plt.xticks(rotation=45)
    plt.show()


def plot_partner_delivery_performance(df):
    """Plot partner on-time delivery."""

    partner = (
        df.groupby("delivery_partner")
        ["on_time_flag"]
        .mean()
        * 100
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=partner.index,
        y=partner.values
    )
    plt.title("Partner Delivery Performance")
    plt.xlabel("Partner")
    plt.ylabel("On-Time %")
    plt.xticks(rotation=45)
    plt.show()


def plot_partner_cost(df):
    """Plot partner costs."""

    partner_cost = (
        df.groupby("delivery_partner")
        ["total_logistics_cost"]
        .mean()
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=partner_cost.index,
        y=partner_cost.values
    )
    plt.title("Partner Cost Comparison")
    plt.xlabel("Partner")
    plt.ylabel("Average Cost")
    plt.xticks(rotation=45)
    plt.show()


def plot_customer_rating_distribution(df):
    """Plot customer ratings."""

    plt.figure(figsize=(8, 5))
    sns.histplot(
        df["customer_rating"],
        bins=5,
        kde=True
    )
    plt.title("Customer Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()


def plot_rating_by_delivery_status(df):
    """Plot rating by delivery status."""

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="delivery_status",
        y="customer_rating"
    )

    plt.title("Rating by Delivery Status")
    plt.xlabel("Delivery Status")
    plt.ylabel("Customer Rating")
    plt.show()
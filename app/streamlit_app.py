import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Logistics Executive & Performance Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .kpi-card {
            padding: 1rem;
            border-radius: 0.75rem;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
            min-height: 120px;
        }

        .kpi-title {
            font-size: 0.85rem;
            color: #6b7280;
            margin-bottom: 0.25rem;
        }

        .kpi-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #111827;
        }

        .kpi-sub {
            font-size: 0.75rem;
            color: #6b7280;
            margin-top: 0.25rem;
        }

        .delta-pos {
            border-left: 4px solid #10b981;
        }

        .delta-neg {
            border-left: 4px solid #ef4444;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Path Configuration & Helper Imports
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.business_analysis import (
        calculate_average_delivery_days,
        calculate_on_time_percentage,
        calculate_total_delayed,
        calculate_total_delivered,
        calculate_total_logistics_cost,
        calculate_total_orders,
    )
except ImportError:
    # Graceful fallback if the dashboard is run independently.
    def calculate_total_orders(df):
        return len(df)

    def calculate_total_delivered(df):
        if "delivery_status" not in df.columns:
            return 0
        return (df["delivery_status"].astype(str).str.lower() == "delivered").sum()

    def calculate_total_delayed(df):
        if "delivery_status" not in df.columns:
            return 0
        return (df["delivery_status"].astype(str).str.lower() == "delayed").sum()

    def calculate_on_time_percentage(df):
        total_orders = calculate_total_orders(df)
        if total_orders == 0:
            return 0.0
        return round(calculate_total_delivered(df) / total_orders * 100, 2)

    def calculate_average_delivery_days(df):
        if "delivery_days" not in df.columns:
            return 0.0
        return round(df["delivery_days"].mean(), 2)

    def calculate_total_logistics_cost(df):
        if "total_logistics_cost" not in df.columns:
            return 0.0
        return df["total_logistics_cost"].sum()


DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "logistics_cleaned.csv")


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
@st.cache_data
def load_dashboard_data():
    """Load and normalize the cleaned logistics dataset."""
    if not os.path.exists(DATA_PATH):
        st.error(f"Cleaned dataset not found: {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    text_columns = [
        "warehouse",
        "shipping_mode",
        "delivery_status",
        "delivery_partner",
        "origin_city",
        "destination_city",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip().str.title()

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    return df


# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------
def render_sidebar_filters(df):
    """Render global dashboard filters and return the filtered dataset."""
    st.sidebar.title("🎛️ Data Slicers")

    filter_keys = [
        "warehouse_filter",
        "origin_filter",
        "destination_filter",
        "shipping_mode_filter",
        "partner_filter",
        "status_filter",
        "date_filter",
    ]

    if st.sidebar.button("Clear all filters", key="clear_filters_button"):
        for key in filter_keys:
            st.session_state.pop(key, None)
        st.rerun()

    st.sidebar.caption("Default view shows all records. Use filters for focused slices.")

    filtered = df.copy()

    # Warehouse filter
    if "warehouse" in df.columns:
        warehouses = sorted(df["warehouse"].dropna().unique().tolist())
        selected_warehouses = st.sidebar.multiselect(
            "Warehouse Location",
            warehouses,
            default=[],
            key="warehouse_filter",
        )
        if selected_warehouses:
            filtered = filtered[filtered["warehouse"].isin(selected_warehouses)]

    # Origin city filter
    if "origin_city" in df.columns:
        origins = sorted(df["origin_city"].dropna().unique().tolist())
        selected_origins = st.sidebar.multiselect(
            "Origin City",
            origins,
            default=[],
            key="origin_filter",
        )
        if selected_origins:
            filtered = filtered[filtered["origin_city"].isin(selected_origins)]

    # Destination city filter
    if "destination_city" in df.columns:
        destinations = sorted(df["destination_city"].dropna().unique().tolist())
        selected_destinations = st.sidebar.multiselect(
            "Destination City",
            destinations,
            default=[],
            key="destination_filter",
        )
        if selected_destinations:
            filtered = filtered[
                filtered["destination_city"].isin(selected_destinations)
            ]

    # Shipping mode filter
    if "shipping_mode" in df.columns:
        shipping_modes = sorted(
            df["shipping_mode"].dropna().unique().tolist()
        )
        selected_modes = st.sidebar.multiselect(
            "Shipping Mode",
            shipping_modes,
            default=[],
            key="shipping_mode_filter",
        )
        if selected_modes:
            filtered = filtered[filtered["shipping_mode"].isin(selected_modes)]

    # Delivery partner filter
    if "delivery_partner" in df.columns:
        partners = sorted(
            df["delivery_partner"].dropna().unique().tolist()
        )
        selected_partners = st.sidebar.multiselect(
            "Delivery Partner",
            partners,
            default=[],
            key="partner_filter",
        )
        if selected_partners:
            filtered = filtered[
                filtered["delivery_partner"].isin(selected_partners)
            ]

    # Delivery status filter
    if "delivery_status" in df.columns:
        statuses = sorted(
            df["delivery_status"].dropna().unique().tolist()
        )
        selected_statuses = st.sidebar.multiselect(
            "Delivery Status",
            statuses,
            default=[],
            key="status_filter",
        )
        if selected_statuses:
            filtered = filtered[
                filtered["delivery_status"].isin(selected_statuses)
            ]

    # Order date filter
    if "order_date" in df.columns and df["order_date"].notna().any():
        min_date = df["order_date"].min().date()
        max_date = df["order_date"].max().date()

        selected_dates = st.sidebar.date_input(
            "Order Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_filter",
        )

        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
            filtered = filtered[
                filtered["order_date"].dt.date.between(start_date, end_date)
            ]

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Showing **{len(filtered):,}** of **{len(df):,}** records"
    )

    return filtered


# ---------------------------------------------------------
# KPI Metrics Row
# ---------------------------------------------------------
def render_kpi_cards(df):
    """Render the executive KPI scorecard."""
    total_orders = calculate_total_orders(df)
    delivered = calculate_total_delivered(df)
    delayed = calculate_total_delayed(df)
    on_time = calculate_on_time_percentage(df)
    avg_days = calculate_average_delivery_days(df)
    total_cost = calculate_total_logistics_cost(df)

    columns = st.columns(6)

    cards = [
        ("Total Orders", f"{total_orders:,}", "All Shipments", "delta-pos"),
        ("Delivered", f"{delivered:,}", "Completed", "delta-pos"),
        ("Delayed", f"{delayed:,}", "Requires Attention", "delta-neg"),
        (
            "On-Time Rate",
            f"{on_time:.1f}%",
            "Target: 85%",
            "delta-pos" if on_time >= 80 else "delta-neg",
        ),
        (
            "Avg Delivery",
            f"{avg_days:.1f} Days",
            "Average Lead Time",
            "delta-pos",
        ),
        (
            "Total Cost",
            f"₹{total_cost:,.0f}",
            "Operating Spend",
            "delta-neg",
        ),
    ]

    for column, (title, value, subtitle, delta_class) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="kpi-card {delta_class}">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------
# View 1: Executive Dashboard
# ---------------------------------------------------------
def render_executive_tab(df):
    if df.empty:
        st.warning("No records match the current filter selections. Please clear one or more filters to restore the data.")
        return

    st.markdown("### Executive Performance Pulse")

    col1, col2 = st.columns(2)

    with col1:
        if "delivery_status" in df.columns:
            status_df = df["delivery_status"].value_counts().reset_index()
            status_df.columns = ["Status", "Shipments"]

            fig_donut = px.pie(
                status_df,
                names="Status",
                values="Shipments",
                hole=0.6,
                title="Delivery Fulfillment Distribution",
                color_discrete_sequence=["#10B981", "#EF4444", "#F59E0B"],
            )
            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent+label",
            )
            fig_donut.update_layout(
                showlegend=True,
                margin=dict(t=40, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        if (
            "shipping_mode" in df.columns
            and "total_logistics_cost" in df.columns
        ):
            mode_cost = (
                df.groupby("shipping_mode")["total_logistics_cost"]
                .agg(["sum", "mean"])
                .reset_index()
            )

            fig_cost = px.bar(
                mode_cost,
                x="shipping_mode",
                y="sum",
                text_auto=".2s",
                title="Total Logistics Cost by Shipping Mode",
                labels={
                    "sum": "Total Logistics Spend",
                    "shipping_mode": "Shipping Mode",
                },
                color="shipping_mode",
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            fig_cost.update_layout(
                showlegend=False,
                margin=dict(t=40, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown("### Top Freight Corridors")

    required_route_columns = {
        "origin_city",
        "destination_city",
        "delivery_status",
    }

    if required_route_columns.issubset(df.columns):
        aggregation = {
            "Shipments": ("delivery_status", "count"),
            "Delayed": (
                "delivery_status",
                lambda values: (
                    values.astype(str).str.lower() == "delayed"
                ).sum(),
            ),
        }

        if "delivery_days" in df.columns:
            aggregation["Avg_Days"] = ("delivery_days", "mean")

        if "total_logistics_cost" in df.columns:
            aggregation["Total_Cost"] = ("total_logistics_cost", "sum")

        routes = (
            df.groupby(["origin_city", "destination_city"])
            .agg(**aggregation)
            .reset_index()
        )

        routes["Shipments"] = pd.to_numeric(
            routes["Shipments"], errors="coerce"
        ).fillna(0)
        routes["Delayed"] = pd.to_numeric(
            routes["Delayed"], errors="coerce"
        ).fillna(0)

        routes["Delay_Rate_%"] = (
            (routes["Delayed"] / routes["Shipments"] * 100)
            .replace([float("inf"), -float("inf")], 0)
            .fillna(0)
        ).round(2)

        if "Avg_Days" in routes.columns:
            routes["Avg_Days"] = routes["Avg_Days"].round(2)

        routes = routes.sort_values("Shipments", ascending=False)

        st.dataframe(
            routes.style.background_gradient(
                subset=["Delay_Rate_%"],
                cmap="Reds",
            ).format(
                {
                    "Total_Cost": "₹{:,.2f}",
                    "Delay_Rate_%": "{:.2f}%",
                }
            ),
            use_container_width=True,
            height=300,
        )


# ---------------------------------------------------------
# View 2: Univariate EDA
# ---------------------------------------------------------
def render_univariate_tab(df):
    if df.empty:
        st.warning("No records match the current filter selections. Please clear one or more filters to restore the data.")
        return

    st.markdown("### Univariate Analysis")

    col1, col2 = st.columns(2)

    with col1:
        categorical_options = [
            column
            for column in [
                "warehouse",
                "shipping_mode",
                "delivery_partner",
                "delivery_status",
            ]
            if column in df.columns
        ]

        if categorical_options:
            selected_category = st.selectbox(
                "Categorical Field",
                categorical_options,
                key="uni_cat",
            )

            counts = df[selected_category].value_counts().reset_index()
            counts.columns = [selected_category, "Count"]

            fig_category = px.bar(
                counts,
                x=selected_category,
                y="Count",
                text="Count",
                title=f"Volume Frequency: {selected_category.title()}",
                color=selected_category,
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_category.update_layout(showlegend=False)
            st.plotly_chart(fig_category, use_container_width=True)

    with col2:
        numeric_options = [
            column
            for column in [
                "delivery_days",
                "total_logistics_cost",
                "distance_km",
                "weight_kg",
            ]
            if column in df.columns
        ]

        if numeric_options:
            selected_numeric = st.selectbox(
                "Numerical Variable",
                numeric_options,
                key="uni_num",
            )

            fig_hist = px.histogram(
                df,
                x=selected_numeric,
                nbins=35,
                marginal="box",
                title=f"Distribution Profile: {selected_numeric.title()}",
                color_discrete_sequence=["#3B82F6"],
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No continuous numeric fields found for histogram.")


# ---------------------------------------------------------
# View 3: Bivariate EDA
# ---------------------------------------------------------
def render_bivariate_tab(df):
    if df.empty:
        st.warning("No records match the current filter selections. Please clear one or more filters to restore the data.")
        return

    st.markdown("### Bivariate Cross-Dimensional Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if "warehouse" in df.columns and "total_logistics_cost" in df.columns:
            fig_box = px.box(
                df,
                x="warehouse",
                y="total_logistics_cost",
                color="warehouse",
                title="Cost Spread Across Warehouse Hubs",
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        if "delivery_partner" in df.columns and "delivery_days" in df.columns:
            partner_performance = (
                df.groupby("delivery_partner")["delivery_days"]
                .mean()
                .reset_index()
                .sort_values("delivery_days")
            )

            fig_partner = px.bar(
                partner_performance,
                x="delivery_days",
                y="delivery_partner",
                orientation="h",
                text_auto=".2f",
                title="Average Delivery Lead Time by Partner",
                color="delivery_days",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_partner, use_container_width=True)

    if "delivery_partner" in df.columns and "delivery_status" in df.columns:
        cross_tab = (
            pd.crosstab(
                df["delivery_partner"],
                df["delivery_status"],
                normalize="index",
            )
            * 100
        ).reset_index()

        status_columns = [
            column
            for column in cross_tab.columns
            if column != "delivery_partner"
        ]

        fig_delay_partner = px.bar(
            cross_tab,
            x="delivery_partner",
            y=status_columns,
            title="Delivery Fulfillment Ratio by Partner (%)",
            barmode="stack",
        )
        st.plotly_chart(fig_delay_partner, use_container_width=True)


# ---------------------------------------------------------
# View 4: Multivariate EDA
# ---------------------------------------------------------
def render_multivariate_tab(df):
    if df.empty:
        st.warning("No records match the current filter selections. Please clear one or more filters to restore the data.")
        return

    st.markdown("### Multivariate Correlation & Multi-Axis Explorer")

    col1, col2 = st.columns(2)

    with col1:
        required_columns = {
            "warehouse",
            "shipping_mode",
            "total_logistics_cost",
        }

        if required_columns.issubset(df.columns):
            pivot = df.pivot_table(
                index="warehouse",
                columns="shipping_mode",
                values="total_logistics_cost",
                aggfunc="mean",
            )

            fig_heat = px.imshow(
                pivot,
                text_auto=".0f",
                aspect="auto",
                title="Cost Matrix: Warehouse vs Shipping Mode (₹ Mean)",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        numeric_columns = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_columns) > 1:
            correlation = df[numeric_columns].corr()

            fig_corr = px.imshow(
                correlation,
                text_auto=".2f",
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu_r",
            )
            st.plotly_chart(fig_corr, use_container_width=True)


# ---------------------------------------------------------
# View 5: Dynamic Self-Service Explorer
# ---------------------------------------------------------
def render_dynamic_explorer(df):
    st.markdown("### Dynamic Self-Service Visualizer")
    st.caption(
        "Build customized visual reports dynamically like in Power BI or Tableau."
    )

    if df.empty:
        st.info("No data is available for the selected filters.")
        return

    column1, column2, column3, column4 = st.columns(4)

    with column1:
        chart_type = st.selectbox(
            "Chart Type",
            [
                "Bar Chart",
                "Scatter Plot",
                "Box Plot",
                "Violin Plot",
                "Treemap",
            ],
        )

    with column2:
        x_axis = st.selectbox(
            "X-Axis Variable",
            df.columns.tolist(),
        )

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    with column3:
        if numeric_columns:
            default_index = min(1, len(numeric_columns) - 1)
            y_axis = st.selectbox(
                "Y-Axis (Metrics)",
                numeric_columns,
                index=default_index,
            )
        else:
            y_axis = None
            st.info("No numeric metric is available.")

    categorical_columns = [
        "None",
        *df.select_dtypes(include=["object", "category"]).columns.tolist(),
    ]

    with column4:
        color_by = st.selectbox(
            "Segment / Color By",
            categorical_columns,
        )

    if y_axis is None:
        return

    color_argument = None if color_by == "None" else color_by

    try:
        if chart_type == "Bar Chart":
            figure = px.histogram(
                df,
                x=x_axis,
                y=y_axis,
                color=color_argument,
                barmode="group",
                title=f"{y_axis} grouped by {x_axis}",
            )

        elif chart_type == "Scatter Plot":
            figure = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=color_argument,
                hover_data=df.columns.tolist()[:4],
                title=f"{y_axis} vs {x_axis}",
            )

        elif chart_type == "Box Plot":
            figure = px.box(
                df,
                x=x_axis,
                y=y_axis,
                color=color_argument,
                title=f"{y_axis} distribution across {x_axis}",
            )

        elif chart_type == "Violin Plot":
            figure = px.violin(
                df,
                x=x_axis,
                y=y_axis,
                color=color_argument,
                box=True,
                title=f"{y_axis} density over {x_axis}",
            )

        else:
            path_columns = [
                x_axis
                if color_argument is None
                else color_argument,
            ]

            if color_argument is not None and color_argument != x_axis:
                path_columns.append(x_axis)

            figure = px.treemap(
                df,
                path=path_columns,
                values=y_axis,
                title=f"Hierarchical Breakdown of {y_axis}",
            )

        figure.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(figure, use_container_width=True)

    except Exception as exc:
        st.error(f"Cannot render the selected dimensions: {exc}")


# ---------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------
def main():
    df = load_dashboard_data()
    filtered_df = render_sidebar_filters(df)

    st.title("🚚 Logistics & Supply Chain Intelligence")
    st.caption(
        "Interactive Executive Operations Monitor & Exploratory Data Platform"
    )

    render_kpi_cards(filtered_df)
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Executive Dashboard",
            "📈 Univariate EDA",
            "🔄 Bivariate Analysis",
            "🌐 Multivariate Matrix",
            "🛠️ Self-Service Builder",
        ]
    )

    with tab1:
        render_executive_tab(filtered_df)

    with tab2:
        render_univariate_tab(filtered_df)

    with tab3:
        render_bivariate_tab(filtered_df)

    with tab4:
        render_multivariate_tab(filtered_df)

    with tab5:
        render_dynamic_explorer(filtered_df)


if __name__ == "__main__":
    main()

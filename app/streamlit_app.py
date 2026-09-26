import base64
import io
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


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


def build_route_summary(df, top_n=5):
    """Build the route summary used in both the dashboard and final report."""
    required_route_columns = {
        "origin_city",
        "destination_city",
        "delivery_status",
    }

    if df.empty or not required_route_columns.issubset(df.columns):
        return pd.DataFrame()

    aggregation = {
        "Shipments": ("delivery_status", "count"),
        "Delayed": (
            "delivery_status",
            lambda values: (values.astype(str).str.lower() == "delayed").sum(),
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

    routes["Shipments"] = pd.to_numeric(routes["Shipments"], errors="coerce").fillna(0)
    routes["Delayed"] = pd.to_numeric(routes["Delayed"], errors="coerce").fillna(0)

    routes["Delay_Rate_%"] = (
        (routes["Delayed"] / routes["Shipments"] * 100)
        .replace([float("inf"), -float("inf")], 0)
        .fillna(0)
    ).round(2)

    if "Avg_Days" in routes.columns:
        routes["Avg_Days"] = routes["Avg_Days"].round(2)

    routes = routes.sort_values(["Shipments", "Delay_Rate_%"], ascending=[False, False])
    return routes.head(top_n).reset_index(drop=True)


def build_chart_snapshot_png(df, chart_type):
    """Create a compact Plotly chart image as PNG bytes for reports and downloads."""
    try:
        if chart_type == "status":
            if "delivery_status" not in df.columns:
                return b""
            status_df = df["delivery_status"].value_counts().reset_index()
            status_df.columns = ["Status", "Shipments"]
            fig = px.pie(
                status_df,
                names="Status",
                values="Shipments",
                hole=0.5,
                color_discrete_sequence=["#10B981", "#EF4444", "#F59E0B"],
            )
        elif chart_type == "shipping_cost":
            if "shipping_mode" not in df.columns or "total_logistics_cost" not in df.columns:
                return b""
            mode_cost = (
                df.groupby("shipping_mode")["total_logistics_cost"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            fig = px.bar(
                mode_cost,
                x="shipping_mode",
                y="total_logistics_cost",
                title="Shipping Cost",
                labels={"shipping_mode": "Shipping Mode", "total_logistics_cost": "Total Cost"},
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
        elif chart_type == "route_delay":
            route_summary = build_route_summary(df, top_n=5)
            if route_summary.empty:
                return b""
            fig = px.bar(
                route_summary,
                x="origin_city",
                y="Delay_Rate_%",
                color="destination_city",
                title="Route Delay Rate",
            )
        else:
            return b""

        fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            width=500,
            height=250,
        )

        image_bytes = fig.to_image(format="png", scale=2)
        return image_bytes
    except Exception:
        return b""


def build_chart_snapshot_image(df, chart_type):
    """Create a compact Plotly chart image as a valid base64 data URI for HTML embedding."""
    image_bytes = build_chart_snapshot_png(df, chart_type)
    if not image_bytes:
        return ""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return "data:image/png;base64," + encoded


def build_report_pdf(df):
    """Generate a real PDF report for the current filtered dataset."""
    if df.empty:
        buffer = io.BytesIO()
        pdf = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph("No data available for report generation.", styles["Title"])]
        pdf.build(story)
        return buffer.getvalue()

    total_orders = calculate_total_orders(df)
    delivered = calculate_total_delivered(df)
    delayed = calculate_total_delayed(df)
    on_time = calculate_on_time_percentage(df)
    avg_days = calculate_average_delivery_days(df)
    total_cost = calculate_total_logistics_cost(df)

    summary_data = [
        ["Total Orders", f"{total_orders:,}"],
        ["Delivered", f"{delivered:,}"],
        ["Delayed", f"{delayed:,}"],
        ["On-Time Rate", f"{on_time:.1f}%"],
        ["Average Delivery Days", f"{avg_days:.1f}"],
        ["Total Logistics Cost", f"₹{total_cost:,.0f}"],
    ]

    route_summary = build_route_summary(df, top_n=5)
    if route_summary.empty:
        route_table = [["No route summary data available."]]
    else:
        route_table = [list(route_summary.columns)] + route_summary.astype(str).values.tolist()

    shipping_data = pd.DataFrame()
    if "shipping_mode" in df.columns and "total_logistics_cost" in df.columns:
        shipping_data = (
            df.groupby("shipping_mode")["total_logistics_cost"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"shipping_mode": "Shipping Mode", "total_logistics_cost": "Total Cost"})
        )
        shipping_data["Total Cost"] = shipping_data["Total Cost"].map(lambda v: f"₹{v:,.0f}")

    if shipping_data.empty:
        shipping_table = [["No shipping-mode summary available."]]
    else:
        shipping_table = [list(shipping_data.columns)] + shipping_data.astype(str).values.tolist()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Logistics Executive Report", styles["Title"]))
    story.append(Paragraph("Generated from the current filtered dataset.", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    summary_table = Table(summary_data, colWidths=[2.5 * inch, 2.5 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    story.append(summary_table)
    story.append(Spacer(1, 0.25 * inch))

    status_png = build_chart_snapshot_png(df, "status")
    shipping_png = build_chart_snapshot_png(df, "shipping_cost")
    route_png = build_chart_snapshot_png(df, "route_delay")

    if status_png:
        story.append(Paragraph("Delivery Status", styles["Heading2"]))
        status_image = Image(io.BytesIO(status_png), width=2.8 * inch, height=2.1 * inch)
        story.append(status_image)

    if shipping_png:
        story.append(Paragraph("Shipping Cost", styles["Heading2"]))
        shipping_image = Image(io.BytesIO(shipping_png), width=2.8 * inch, height=2.1 * inch)
        story.append(shipping_image)

    story.append(Paragraph("Top Freight Corridors", styles["Heading2"]))
    route_table_obj = Table(route_table, colWidths=[1.2 * inch, 1.2 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 1.1 * inch])
    route_table_obj.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(route_table_obj)

    if route_png:
        story.append(Spacer(1, 0.2 * inch))
        story.append(Image(io.BytesIO(route_png), width=4.5 * inch, height=2.2 * inch))

    story.append(Paragraph("Cost by Shipping Mode", styles["Heading2"]))
    shipping_table_obj = Table(shipping_table, colWidths=[2.5 * inch, 2.5 * inch])
    shipping_table_obj.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(shipping_table_obj)

    doc.build(story)
    return buffer.getvalue()


def build_report_html(df):
    """Generate a presentation-ready HTML executive report from the current filtered dataset."""
    if df.empty:
        return "<html><body><h2>No data available for report generation.</h2></body></html>"

    total_orders = calculate_total_orders(df)
    delivered = calculate_total_delivered(df)
    delayed = calculate_total_delayed(df)
    on_time = calculate_on_time_percentage(df)
    avg_days = calculate_average_delivery_days(df)
    total_cost = calculate_total_logistics_cost(df)

    route_summary = build_route_summary(df, top_n=5)
    if route_summary.empty:
        route_table_html = "<p>No route summary data available.</p>"
    else:
        route_table_html = route_summary.to_html(index=False, border=0, justify="left")

    shipping_mode_cost = pd.DataFrame()
    if "shipping_mode" in df.columns and "total_logistics_cost" in df.columns:
        shipping_mode_cost = (
            df.groupby("shipping_mode")["total_logistics_cost"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"shipping_mode": "Shipping Mode", "total_logistics_cost": "Total Cost"})
        )
        shipping_mode_cost["Total Cost"] = shipping_mode_cost["Total Cost"].map("₹{:,.0f}".format)

    shipping_table_html = (
        shipping_mode_cost.to_html(index=False, border=0, justify="left")
        if not shipping_mode_cost.empty
        else "<p>No shipping-mode summary available.</p>"
    )

    summary_rows = [
        ("Total Orders", f"{total_orders:,}"),
        ("Delivered", f"{delivered:,}"),
        ("Delayed", f"{delayed:,}"),
        ("On-Time Rate", f"{on_time:.1f}%"),
        ("Average Delivery Days", f"{avg_days:.1f}"),
        ("Total Logistics Cost", f"₹{total_cost:,.0f}"),
    ]

    summary_html = "".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in summary_rows
    )

    status_img = build_chart_snapshot_image(df, "status")
    shipping_img = build_chart_snapshot_image(df, "shipping_cost")
    route_img = build_chart_snapshot_image(df, "route_delay")

    status_img_html = (
        f'<img src="{status_img}" style="max-width: 100%; height: auto; border: 1px solid #e5e7eb;" />'
        if status_img
        else "<p>No delivery-status chart available.</p>"
    )
    shipping_img_html = (
        f'<img src="{shipping_img}" style="max-width: 100%; height: auto; border: 1px solid #e5e7eb;" />'
        if shipping_img
        else "<p>No shipping-cost chart available.</p>"
    )
    route_img_html = (
        f'<img src="{route_img}" style="max-width: 100%; height: auto; border: 1px solid #e5e7eb;" />'
        if route_img
        else "<p>No route-delay chart available.</p>"
    )

    html = f"""
    <html>
      <head>
        <title>Logistics Executive Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; }}
          h1, h2 {{ color: #111827; }}
          table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
          th, td {{ padding: 10px 12px; border: 1px solid #e5e7eb; text-align: left; }}
          th {{ background: #f3f4f6; }}
          .summary {{ width: 60%; margin-bottom: 18px; }}
          .charts {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 12px; }}
          .chart-box {{ width: 48%; min-width: 300px; }}
        </style>
      </head>
      <body>
        <h1>Logistics Executive Report</h1>
        <p>Generated from the current filtered dataset.</p>

        <h2>Executive Summary</h2>
        <table class="summary">
          <tbody>
            {summary_html}
          </tbody>
        </table>

        <div class="charts">
          <div class="chart-box">
            <h3>Delivery Status</h3>
            {status_img_html}
          </div>
          <div class="chart-box">
            <h3>Shipping Cost</h3>
            {shipping_img_html}
          </div>
        </div>

        <h2>Top Freight Corridors</h2>
        {route_table_html}
        {route_img_html}

        <h2>Cost by Shipping Mode</h2>
        {shipping_table_html}
      </body>
    </html>
    """
    return html


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

    report_html = build_report_html(filtered_df)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Executive Report")
    st.sidebar.download_button(
        label="Download HTML Report",
        data=report_html,
        file_name="logistics_executive_report.html",
        mime="text/html",
    )
    report_pdf = build_report_pdf(filtered_df)
    st.sidebar.download_button(
        label="Download PDF Report",
        data=report_pdf,
        file_name="logistics_executive_report.pdf",
        mime="application/pdf",
    )

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

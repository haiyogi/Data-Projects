import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dashboard.charts import (
    category_pareto_chart,
    category_treemap,
    delay_distribution_chart,
    delay_severity_chart,
    delivery_status_chart,
    market_chart,
    quarterly_trend_chart,
    segment_status_chart,
    shipping_mode_chart,
    yearly_trend_chart,
)
from src.config import S3_BUCKET, S3_OBJECT_KEY
from src.data_loader import load_from_s3
from src.database import MongoRepository
from src.preprocessing import clean_data

st.set_page_config(page_title="Supply Chain Delivery Monitor", layout="wide")
st.title("Supply Chain Delivery Monitor")
st.caption("DataCo SMART SUPPLY CHAIN | Delivery performance dashboard")


@st.cache_data
def get_data():
    # Cache the S3 dataset so filters stay responsive
    return clean_data(load_from_s3(S3_BUCKET, S3_OBJECT_KEY))


df = get_data()
mongo = MongoRepository()

with st.sidebar:
    #  the main page  operational filters
    st.header("Filters")
    markets = st.multiselect("Market", sorted(df["Market"].dropna().unique()))
    shipping_modes = st.multiselect("Shipping Mode", sorted(df["Shipping Mode"].dropna().unique()))

filtered = df.copy()
# Apply only the filters selected
if markets:
    filtered = filtered[filtered["Market"].isin(markets)]
if shipping_modes:
    filtered = filtered[filtered["Shipping Mode"].isin(shipping_modes)]

# Keep dashboard sections separate for easier reading
overview_tab, group_tab, trend_tab, delay_tab, modeling_tab = st.tabs([
    "Dataset Overview", "Group Analysis", "Delivery Trends",
    "Category & Delay Analysis", "Modeling",
])

with overview_tab:
    st.subheader("Dataset Overview")
    kpis = st.columns(4)
    # Show the main dataset facts without charts
    kpis[0].metric("Orders", f"{len(filtered):,}")
    kpis[1].metric("Late Delivery Rate", f"{filtered['Late_delivery_risk'].mean() * 100:.2f}%")
    kpis[2].metric("Average Sales", f"{filtered['Sales'].mean():,.2f}")
    kpis[3].metric("Average Profit", f"{filtered['Order Profit Per Order'].mean():,.2f}")

    overview = pd.DataFrame({
        "Metric": [
            "Rows", "Columns", "Unique Orders", "Unique Customers",
            "First Order Date", "Last Order Date",
        ],
        "Value": [
            f"{len(filtered):,}", f"{filtered.shape[1]:,}",
            f"{filtered['Order Id'].nunique():,}",
            f"{filtered['Customer Id'].nunique():,}",
            filtered["order date (DateOrders)"].min().strftime("%Y-%m-%d"),
            filtered["order date (DateOrders)"].max().strftime("%Y-%m-%d"),
        ],
    })
    st.dataframe(overview, use_container_width=True, hide_index=True)
    st.subheader("Dataset Sample")
    st.dataframe(filtered.head(10), use_container_width=True, hide_index=True)

with group_tab:
    st.subheader("Group Analysis")
    # Keep the required delivery-status donut chart
    st.plotly_chart(delivery_status_chart(filtered), use_container_width=True)
    st.plotly_chart(shipping_mode_chart(filtered), use_container_width=True)
    st.plotly_chart(market_chart(filtered), use_container_width=True)
    st.plotly_chart(segment_status_chart(filtered), use_container_width=True)

with trend_tab:
    st.subheader("Delivery Trends")
    st.plotly_chart(quarterly_trend_chart(filtered), use_container_width=True)
    st.plotly_chart(yearly_trend_chart(filtered), use_container_width=True)

with delay_tab:
    st.subheader("Category and Delay Analysis")
    st.plotly_chart(category_treemap(filtered), use_container_width=True)
    st.plotly_chart(category_pareto_chart(filtered), use_container_width=True)
    st.plotly_chart(delay_severity_chart(filtered), use_container_width=True)
    st.plotly_chart(delay_distribution_chart(filtered), use_container_width=True)

with modeling_tab:
    st.subheader("Model Performance")
    if mongo.check_connection():
        metrics = mongo.latest_metrics()
        if metrics:
            metric_rows = [
                {
                    "Model": name.replace("_", " ").title(),
                    "Accuracy (%)": values["accuracy"],
                    "Precision (%)": values["precision"],
                    "Recall (%)": values["recall"],
                    "F1-score (%)": values["f1_score"],
                    "ROC-AUC (%)": values["roc_auc"],
                }
                for name, values in metrics.items()
            ]
            st.dataframe(metric_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No model metrics found. Run `python run_pipeline.py` first.")
    else:
        st.info("MongoDB is not connected. Start MongoDB and run `python run_pipeline.py` first.")

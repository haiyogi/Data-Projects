import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def delivery_status_chart(df):
    # Summarise the four delivery outcomes
    counts = df["Delivery Status"].value_counts().reset_index()
    counts.columns = ["Delivery Status", "Count"]
    fig = px.pie(counts, names="Delivery Status", values="Count", hole=0.5,
                 title="Overall Delivery Status Distribution",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(title_x=0.5, title_font_size=20, height=450, annotations=[{
        "text": f"<b>{len(df):,}</b><br>Total Orders",
        "x": 0.5, "y": 0.5, "showarrow": False,
    }])
    return fig


def shipping_mode_chart(df):
    # Compare late-delivery percentages by shipping method
    rates = df.groupby("Shipping Mode")["Late_delivery_risk"].mean().mul(100).reset_index()
    rates.columns = ["Shipping Mode", "Late Delivery Rate"]
    rates = rates.sort_values("Late Delivery Rate", ascending=False)
    fig = px.bar(rates, x="Shipping Mode", y="Late Delivery Rate",
                 color="Late Delivery Rate", text="Late Delivery Rate",
                 color_continuous_scale="Turbo",
                 title="Late Delivery Rate by Shipping Mode")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        title_x=0.5, title_font_size=20, height=450,
        yaxis_ticksuffix="%", coloraxis_showscale=False,
        margin=dict(t=80, b=70, l=70, r=70),
    )
    return fig


def market_chart(df):
    # Add order counts 
    rates = df.groupby("Market")["Late_delivery_risk"].agg(["mean", "count"]).reset_index()
    rates["Late Delivery Rate"] = rates["mean"] * 100
    rates = rates.sort_values("Late Delivery Rate")
    fig = px.bar(rates, x="Late Delivery Rate", y="Market", orientation="h",
                 text="Late Delivery Rate", color="Late Delivery Rate",
                 color_continuous_scale="Viridis",
                 title="Late Delivery Rate by Global Market",
                 custom_data=["count"])
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        title_x=0.5, title_font_size=20, height=500,
        xaxis_ticksuffix="%", coloraxis_showscale=False,
        margin=dict(t=80, b=70, l=120, r=120),
    )
    fig.update_xaxes(
        range=[0, rates["Late Delivery Rate"].max() * 1.15],
        gridcolor="lightgrey",
        automargin=True,
    )
    fig.update_yaxes(automargin=True)
    return fig


def category_treemap(df):

    data = df[df["Late_delivery_risk"] == 1]["Category Name"].value_counts().reset_index()
    data.columns = ["Category Name", "Late Deliveries"]
    fig = px.treemap(data.head(15), path=["Category Name"], values="Late Deliveries",
                     color="Late Deliveries", color_continuous_scale="Turbo",
                     title="Late Deliveries by Product Category")
    fig.update_traces(texttemplate="<b>%{label}</b><br>%{value:,} Late Deliveries")
    fig.update_layout(
        title_x=0.5, title_font_size=20, height=650,
        coloraxis_showscale=False,
        margin=dict(t=90, b=30, l=30, r=30),
    )
    return fig


def quarterly_trend_chart(df):
    # Compare quarter performance across all years
    data = df.copy()
    data["Quarter"] = "Q" + data["order date (DateOrders)"].dt.quarter.astype(str)
    quarterly = data.groupby("Quarter")["Late_delivery_risk"].mean().mul(100).reset_index()
    quarterly.columns = ["Quarter", "Late Delivery Rate"]
    quarterly["Quarter"] = pd.Categorical(
        quarterly["Quarter"], categories=["Q1", "Q2", "Q3", "Q4"], ordered=True
    )
    quarterly = quarterly.sort_values("Quarter")

    fig = px.line(
        quarterly, x="Quarter", y="Late Delivery Rate", markers=True,
        text="Late Delivery Rate", title="Average Late Delivery Rate by Quarter",
    )
    fig.update_traces(
        line=dict(width=4), marker=dict(size=9),
        texttemplate="%{text:.2f}%", textposition="top center",
        hovertemplate="<b>%{x}</b><br>Late Delivery Rate: %{y:.2f}%<extra></extra>",
    )
    fig.update_layout(
        title_x=0.5, title_font_size=20, height=500,
        yaxis_title="Late Delivery Rate (%)", yaxis_ticksuffix="%",
        margin=dict(t=80, b=70, l=70, r=70),
    )
    return fig


def yearly_trend_chart(df):
    # Track average late-delivery rate by year
    data = df.copy()
    data["Year"] = data["order date (DateOrders)"].dt.year.astype(str)
    yearly = data.groupby("Year")["Late_delivery_risk"].mean().mul(100).reset_index()
    yearly.columns = ["Year", "Late Delivery Rate"]

    fig = px.line(
        yearly, x="Year", y="Late Delivery Rate", markers=True,
        text="Late Delivery Rate", title="Yearly Average Late Delivery Rate",
    )
    fig.update_traces(
        line=dict(width=4), marker=dict(size=9),
        texttemplate="%{text:.2f}%", textposition="top center",
        hovertemplate="<b>%{x}</b><br>Late Delivery Rate: %{y:.2f}%<extra></extra>",
    )
    fig.update_layout(
        title_x=0.5, title_font_size=20, height=500,
        xaxis_title="Year", yaxis_title="Late Delivery Rate (%)",
        yaxis_ticksuffix="%", margin=dict(t=80, b=70, l=70, r=70),
    )
    # Keep only actual years on the axis
    fig.update_xaxes(type="category")
    return fig


def category_pareto_chart(df):
    # Show the categories contributing most late deliveries
    data = df[df["Late_delivery_risk"] == 1]["Category Name"].value_counts().reset_index()
    data.columns = ["Category Name", "Late Deliveries"]
    data["Cumulative Percentage"] = (
        data["Late Deliveries"].cumsum() / data["Late Deliveries"].sum() * 100
    )
    data = data.head(15)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data["Category Name"], y=data["Late Deliveries"],
        name="Late Deliveries", text=data["Late Deliveries"],
        texttemplate="%{text:,}", textposition="outside",
        marker=dict(color=data["Late Deliveries"], colorscale="Turbo"),
        hovertemplate="<b>%{x}</b><br>Late Deliveries: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=data["Category Name"], y=data["Cumulative Percentage"],
        name="Cumulative %", yaxis="y2", mode="lines+markers",
        line=dict(width=3), marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(
        y=80, yref="y2", line_dash="dash", line_width=2,
        annotation_text="80% Reference", annotation_position="right",
    )
    fig.update_layout(
        title="Pareto Analysis of Late Deliveries by Product Category",
        title_x=0.5, title_font_size=20, height=600,
        xaxis=dict(title="Product Category", tickangle=-45),
        yaxis=dict(title="Number of Late Deliveries"),
        yaxis2=dict(title="Cumulative Percentage", overlaying="y", side="right",
                    range=[0, 105], ticksuffix="%"),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        margin=dict(t=110, b=170, l=80, r=90),
    )
    return fig


def segment_status_chart(df):
    # Compare delivery outcomes across customer segments
    data = pd.crosstab(
        df["Customer Segment"], df["Delivery Status"], normalize="index"
    ).mul(100).reset_index()
    data = data.melt(
        id_vars="Customer Segment", var_name="Delivery Status", value_name="Percentage"
    )
    fig = px.bar(
        data, x="Customer Segment", y="Percentage", color="Delivery Status",
        text="Percentage", barmode="stack", title="Delivery Performance by Customer Segment",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%", textposition="inside",
        hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.2f}%<extra></extra>",
    )
    fig.update_layout(
        title_x=0.5, title_font_size=20, height=550,
        yaxis_title="Percentage of Orders", yaxis_ticksuffix="%",
        yaxis_range=[0, 100], legend_title="Delivery Status",
        margin=dict(t=90, b=70, l=70, r=70),
    )
    return fig


def delay_severity_chart(df):
    # Measure average delay among late shipments
    data = df.copy()
    data["Delay Days"] = (
        data["Days for shipping (real)"] - data["Days for shipment (scheduled)"]
    )
    summary = (
        data[data["Delay Days"] > 0]
        .groupby("Shipping Mode")
        .agg(Average_Delay=("Delay Days", "mean"), Late_Orders=("Delay Days", "count"))
        .reset_index().sort_values("Average_Delay", ascending=False)
    )
    fig = px.bar(
        summary, x="Shipping Mode", y="Average_Delay", color="Average_Delay",
        text="Average_Delay", custom_data=["Late_Orders"],
        color_continuous_scale="Sunsetdark", title="Average Delay Severity by Shipping Mode",
    )
    fig.update_traces(
        texttemplate="%{text:.2f} days", textposition="outside",
        hovertemplate="<b>%{x}</b><br>Average Delay: %{y:.2f} days<br>"
                      "Late Orders: %{customdata[0]:,}<extra></extra>",
    )
    fig.update_layout(
        title_x=0.5, title_font_size=20, height=500,
        yaxis_title="Average Delay (Days)", coloraxis_showscale=False,
        margin=dict(t=90, b=80, l=80, r=70),
    )
    fig.update_yaxes(range=[0, summary["Average_Delay"].max() * 1.2], gridcolor="lightgrey")
    return fig


def delay_distribution_chart(df):
    # Display actual shipping time minus scheduled shipping time
    data = df.copy()
    data["Shipping Delay Days"] = (
        data["Days for shipping (real)"] - data["Days for shipment (scheduled)"]
    )
    distribution = data["Shipping Delay Days"].value_counts().sort_index().reset_index()
    distribution.columns = ["Shipping Delay Days", "Number of Orders"]

    fig = px.bar(
        distribution, x="Shipping Delay Days", y="Number of Orders",
        color="Shipping Delay Days", text="Number of Orders",
        color_continuous_scale="RdYlGn_r", title="Distribution of Actual Shipping Delay Duration",
    )
    fig.update_traces(
        texttemplate="%{text:,}", textposition="outside",
        hovertemplate="<b>Delay: %{x} days</b><br>Orders: %{y:,}<extra></extra>",
    )
    fig.add_vline(
        x=0, line_dash="dash", line_width=3,
        annotation_text="Scheduled Delivery", annotation_position="top",
    )
    fig.update_layout(
        title_x=0.5, title_font_size=20, height=550,
        xaxis_title="Actual Shipping Days - Scheduled Shipping Days",
        yaxis_title="Number of Orders", coloraxis_showscale=False,
        margin=dict(t=90, b=70, l=80, r=70),
    )
    return fig

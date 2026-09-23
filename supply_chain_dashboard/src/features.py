import math

import pandas as pd

LEAKAGE_COLUMNS = {
    "Late_delivery_risk", "Delivery Status", "Days for shipping (real)",
    "shipping date (DateOrders)",
}


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    order_date = result["order date (DateOrders)"]
    # Turn the order timestamp into useful model features
    result["order_year"] = order_date.dt.year
    result["order_month"] = order_date.dt.month
    result["order_day_of_week"] = order_date.dt.dayofweek
    result["order_hour"] = order_date.dt.hour
    result["is_weekend"] = (order_date.dt.dayofweek >= 5).astype(int)
    result["month_sin"] = (result["order_month"] * 2 * math.pi / 12).map(math.sin)
    result["month_cos"] = (result["order_month"] * 2 * math.pi / 12).map(math.cos)
    result["order_value"] = result["Order Item Total"].fillna(result["Sales"])
    result["discount_amount"] = result["Order Item Discount"].fillna(0)
    result["discount_rate"] = result["Order Item Discount Rate"].fillna(0)
    result["price_quantity_value"] = (
        result["Order Item Product Price"] * result["Order Item Quantity"]
    )

    # Add profitability category for modeling
    result["Profit Category"] = result["Benefit per order"].apply(
        lambda value: (
            "Positive Profit" if value >= 0 else "Negative Profit"
        )
        if pd.notna(value) else "Unknown Profit"
    )
    return result


def make_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build features that are available before delivery."""
    result = add_date_features(df)
    # Remove fields that reveal the delivery outcome
    result = result.drop(columns=list(LEAKAGE_COLUMNS), errors="ignore")
    result = result.drop(columns=["order date (DateOrders)"], errors="ignore")
    # Drop identifiers and free text that do not generalise well
    remove = [
        "Order Id", "Order Item Id", "Customer Id", "Order Customer Id",
        "Product Image", "Product Name", "Customer City", "Order City",
    ]
    result = result.drop(columns=remove, errors="ignore")
    return result, df["Late_delivery_risk"].astype(int)

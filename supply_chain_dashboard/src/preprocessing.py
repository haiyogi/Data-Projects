import pandas as pd


DATE_COLUMNS = ["order date (DateOrders)", "shipping date (DateOrders)"]

REQUIRED_COLUMNS = [
    "Order Id", "Late_delivery_risk", "Delivery Status", "Shipping Mode",
    "Market", "Sales", "Order Profit Per Order", "order date (DateOrders)",
]

NUMERIC_COLUMNS = [
    "Days for shipping (real)", "Days for shipment (scheduled)",
    "Benefit per order", "Sales per customer", "Late_delivery_risk",
    "Category Id", "Customer Id", "Customer Zipcode", "Department Id",
    "Latitude", "Longitude", "Order Customer Id", "Order Id",
    "Order Item Discount", "Order Item Discount Rate", "Order Item Id",
    "Order Item Product Price", "Order Item Profit Ratio", "Order Item Quantity",
    "Sales", "Order Item Total", "Order Profit Per Order", "Order Zipcode",
    "Product Card Id", "Product Category Id", "Product Price", "Product Status",
]

OUTLIER_EXCLUDE_COLUMNS = {
    "Late_delivery_risk", "Order Id", "Order Item Id", "Customer Id",
    "Order Customer Id", "Category Id", "Department Id", "Product Card Id",
    "Product Category Id", "Customer Zipcode", "Order Zipcode",
    "Latitude", "Longitude", "Product Status",
    "Days for shipping (real)", "Days for shipment (scheduled)",
}


def handle_outliers(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """Clip extreme numeric values using IQR limits"""
    clean = df.copy()
    numeric_columns = clean.select_dtypes(include="number").columns

    for column in numeric_columns:
        # Keep identifiers, flags, and operational day fields unchanged
        if column in OUTLIER_EXCLUDE_COLUMNS:
            continue
        if clean[column].nunique(dropna=True) <= 2:
            continue

        first_quartile = clean[column].quantile(0.25)
        third_quartile = clean[column].quantile(0.75)
        iqr = third_quartile - first_quartile
        if pd.isna(iqr) or iqr == 0:
            continue

        lower_limit = first_quartile - multiplier * iqr
        upper_limit = third_quartile + multiplier * iqr
        clean[column] = clean[column].clip(lower_limit, upper_limit)

    return clean


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the raw DataCo supply-chain data"""
    # Work on a copy
    clean = df.copy()
    clean.columns = clean.columns.astype(str).str.strip()

    # Stop early if the source schema is incomplete
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(clean.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    # Remove fully duplicated rows before analysis
    clean = clean.drop_duplicates().reset_index(drop=True)

    # Standardise text fields and turn blank values into missing values
    text_columns = clean.select_dtypes(include="object").columns
    for column in text_columns:
        clean[column] = clean[column].str.strip()
        clean[column] = clean[column].replace({"": pd.NA, "nan": pd.NA})

    # Convert known numeric fields
    for column in NUMERIC_COLUMNS:
        if column in clean:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")

    # Convert date text before feature creation
    for column in DATE_COLUMNS:
        if column in clean:
            clean[column] = pd.to_datetime(clean[column], errors="coerce")

    # Replace infinite values created by source calculations
    numeric_columns = clean.select_dtypes(include="number").columns
    clean[numeric_columns] = clean[numeric_columns].replace(
        [float("inf"), -float("inf")], pd.NA
    )

    # Keep only valid binary target values
    clean = clean[clean["Late_delivery_risk"].isin([0, 1])].copy()

    # Negative quantities and shipping durations are not valid
    for column in [
        "Order Item Quantity", "Days for shipping (real)",
        "Days for shipment (scheduled)",
    ]:
        if column in clean:
            clean.loc[clean[column] < 0, column] = pd.NA

    # Clip extreme financial and quantity values without deleting rows
    clean = handle_outliers(clean)

    # An order date is required for time-based analysis
    clean = clean.dropna(subset=["order date (DateOrders)"])

    # Remove sensitive fields and columns with very little usable data
    drop_columns = [
        "Product Description", "Customer Email", "Customer Password",
        "Customer Fname", "Customer Lname", "Customer Street",
        "Order Zipcode",
    ]
    clean = clean.drop(columns=drop_columns, errors="ignore")

    return clean.reset_index(drop=True)

from src.config import S3_BUCKET, S3_OBJECT_KEY
from src.data_loader import load_from_s3
from src.database import MongoRepository
from src.features import make_model_data
from src.modeling import train_models
from src.preprocessing import clean_data


def build_analytics(df):
    """Create dashboard summaries"""
    shipping = (
        df.groupby("Shipping Mode")["Late_delivery_risk"]
        .mean().mul(100).round(2).to_dict()
    )
    markets = (
        df.groupby("Market")["Late_delivery_risk"]
        .mean().mul(100).round(2).to_dict()
    )
    return {
        "total_records": int(len(df)),
        "late_delivery_rate": round(float(df["Late_delivery_risk"].mean() * 100), 2),
        "average_sales": round(float(df["Sales"].mean()), 2),
        "average_profit": round(float(df["Order Profit Per Order"].mean()), 2),
        "shipping_mode_late_rate": shipping,
        "market_late_rate": markets,
    }


def main():
    # Load and clean the source dataset
    df = clean_data(load_from_s3(S3_BUCKET, S3_OBJECT_KEY))

    # MongoDB is required for the pipeline output
    mongo = MongoRepository()
    if not mongo.check_connection():
        raise ConnectionError(
            "MongoDB is unavailable. Start MongoDB and run the pipeline again"
        )

    # Prepare leakage-safe model inputs
    features, target = make_model_data(df)
    metrics, fitted_models = train_models(features, target)

    # Store the fields needed by the dashboard and database users
    processed_columns = [
        "Order Id", "Market", "Shipping Mode", "Category Name",
        "Delivery Status", "Late_delivery_risk", "Sales",
        "Order Profit Per Order", "order date (DateOrders)",
    ]
    mongo.save_dataframe("processed_orders", df[processed_columns])

    # Use the model with the highest ROC-AUC for risk scores
    best_name = max(metrics, key=lambda name: metrics[name]["roc_auc"])
    best_model = fitted_models[best_name]
    probabilities = best_model.predict_proba(features)[:, 1]
    predictions = df[[
        "Order Id", "Market", "Shipping Mode", "Category Name",
        "order date (DateOrders)",
    ]].copy()
    predictions["predicted_late_delivery"] = (probabilities >= 0.5).astype(int)
    predictions["prediction_probability"] = probabilities.round(4)
    # Store predictions
    mongo.save_predictions(predictions)
    mongo.save_metrics(metrics)
    mongo.save_summary(build_analytics(df))
    print("Processed data, predictions, metrics, and analytics saved to MongoDB.")

    for name, values in metrics.items():
        print(name, {key: round(value, 4) for key, value in values.items()})


if __name__ == "__main__":
    main()

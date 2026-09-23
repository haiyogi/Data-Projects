# DataCo Supply Chain Delivery Platform

An end-to-end supply-chain analytics platform for analysing and predicting late deliveries from the DataCo SMART SUPPLY CHAIN dataset.

The project combines AWS S3 ingestion, Pandas preprocessing, feature engineering, three machine-learning models, MongoDB storage, and a Streamlit dashboard.

Live output : https://haiyogi-data-projects-supply-chain-dashboarddashboardapp-xyknfb.streamlit.app/

## Project objective

The system identifies historical delivery patterns and predicts whether an order is likely to be delivered late. It also provides interactive analysis by market, shipping mode, customer segment, product category, and time period.

## Data source

The active dataset is loaded directly from:

```text
s3://supply-chain-dataset-bucket/DataCoSupplyChainDataset.csv
```

The S3 location is configured in `src/config.py`.

## Project structure

```text
data_co_proosal/
├── dashboard/
│   ├── app.py
│   ├── charts.py
│   └── __init__.py
├── models/
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── lightgbm.joblib
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── modeling.py
│   ├── database.py
│   └── __init__.py
├── run_pipeline.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
└── README.md
```

## Code responsibilities

### `src/config.py`

Stores the S3 bucket, S3 object key, model directory, and reproducibility seed.

### `src/data_loader.py`

Downloads the CSV from S3 using `boto3` and loads it into a Pandas DataFrame.

### `src/preprocessing.py`

Validates required columns, removes duplicate rows, cleans text values, converts numeric and date columns, handles invalid values, clips numerical outliers using IQR limits, and removes sensitive or highly incomplete fields.

### `src/features.py`

Creates date, seasonal, order-value, discount, price-quantity, weekend, and profit-category features. Delivery-outcome fields are removed to prevent target leakage.

### `src/modeling.py`

Builds the machine-learning pipeline using:

1. Missing-value imputation
2. Ordinal encoding for categorical data
3. Min-Max scaling fitted only on training data
4. Random Forest, XGBoost, and LightGBM classifiers
5. Weighted evaluation metrics
6. Standard `0.5` prediction threshold
7. Joblib model persistence in `models/`

Metrics include Accuracy, Precision, Recall, F1-score, and ROC-AUC. Precision, Recall, and F1-score use weighted averages across both classes.

### `src/database.py`

Provides the MongoDB repository for connection checks and storage of processed records, predictions, model metrics, and analytical summaries. Large DataFrames are inserted in batches.

### `run_pipeline.py`

Runs the complete backend workflow:

```text
AWS S3 → preprocessing → feature engineering → model training
      → prediction generation → MongoDB storage
```

MongoDB is required. The pipeline stops before model training if MongoDB is unavailable.

### `dashboard/charts.py`

Contains reusable Plotly chart functions. Keeping chart logic here prevents duplication in the dashboard application.

### `dashboard/app.py`

Loads the S3 dataset, applies preprocessing, provides filters, displays dashboard tabs, and reads the latest model metrics from MongoDB.

## MongoDB collections

### `processed_orders`

Stores cleaned order fields used by the dashboard and database users.

### `predictions`

Stores order details, predicted late-delivery class, and prediction probability.

### `model_metrics`

Stores timestamped evaluation results for Random Forest, XGBoost, and LightGBM.

### `dashboard_summaries`

Stores total records, late-delivery rates, average sales, average profit, and grouped rates by shipping mode and market.

## Dashboard tabs

### Dataset Overview

Shows dataset KPIs, row and column counts, unique orders and customers, date range, and a sample of the dataset.

### Group Analysis

Shows the delivery-status donut chart, late-delivery rate by shipping mode, market analysis, and customer-segment delivery performance.

### Delivery Trends

Shows average late-delivery rate by quarter (`Q1`–`Q4`) and yearly average late-delivery rate.

### Category & Delay Analysis

Shows the product-category treemap, product-category Pareto analysis, average delay severity by shipping mode, and delay-duration distribution.

### Modeling

Shows the latest weighted model metrics retrieved from MongoDB.

## Installation

```bash
pip install -r requirements.txt
```

The main dependencies are Pandas, NumPy, scikit-learn, XGBoost, LightGBM, boto3, PyMongo, Plotly, Streamlit, Joblib, and python-dotenv.

## Environment configuration

Create `.env` using `.env.example`:

```env
MONGODB_URI=mongodb_connection_string
MONGODB_DATABASE=supply_chain
```

Configure AWS credentials through the boto3 credential chain:

```bash
aws configure
```

Or set AWS environment variables:

```env
AWS_ACCESS_KEY_ID=access_key
AWS_SECRET_ACCESS_KEY=secret_key
AWS_DEFAULT_REGION=region
```

Never commit `.env` or cloud credentials.

## Run the pipeline

Ensure AWS S3 and MongoDB are available, then run:

```bash
python run_pipeline.py
```

The pipeline downloads the dataset from S3, trains all three models, saves model files in `models/`, and writes processed data, predictions, metrics, and analytics to MongoDB.

## Run the dashboard

After the pipeline completes:

```bash
streamlit run dashboard/app.py
```

```bash
pip install -r requirements.txt
```

## Reproducibility and leakage controls

- A fixed random seed is used for the train-test split
- Delivery outcome fields are excluded from model features
- Min-Max scaling is fitted inside the training pipeline
- Unknown categorical values are encoded safely as `-1`
- Sensitive customer fields are excluded from stored processed outputs

## Conclusion

This project provides a complete supply-chain delivery analytics solution using AWS S3, Python-based preprocessing, machine learning, MongoDB, and Streamlit. The platform transforms the DataCo dataset into useful delivery insights and predicts late-delivery risk using Random Forest, XGBoost, and LightGBM. The dashboard supports operational monitoring through interactive visualisations, while MongoDB provides central storage for processed records, predictions, model metrics, and analytical outputs. The modular structure also allows the system to be extended later with new data sources, features, models, and monitoring capabilities.

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .config import MODEL_DIR, RANDOM_STATE


def numeric_columns(data):
    """Select numeric columns for the preprocessing pipeline"""
    return data.select_dtypes(exclude=["object", "category"]).columns


def categorical_columns(data):
    """Select text and category columns for the preprocessing pipeline"""
    return data.select_dtypes(include=["object", "category"]).columns


def build_pipeline(model) -> Pipeline:
    # Impute missing values and encode categories inside the pipeline
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"),
         numeric_columns),
        ("categorical", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            # Use -1 for categories not seen during training
            ("encode", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )),
        ]), categorical_columns),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        # Fit scaling only on the training split through the pipeline
        ("scaler", MinMaxScaler()),
        ("model", model),
    ])


def train_models(X: pd.DataFrame, y: pd.Series) -> tuple[dict, dict]:
    # Use a stratified split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    # Compare three tree-based classifiers
    models = {
        "random_forest": build_pipeline(RandomForestClassifier(
            n_estimators=500, n_jobs=-1,max_depth=50 ,
            random_state=RANDOM_STATE, class_weight="balanced"
        )),
        "xgboost": build_pipeline(XGBClassifier(
            n_estimators=500, max_depth=25, learning_rate=0.05,
            min_child_weight=3, reg_lambda=2,
            eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=4, subsample=0.9, colsample_bytree=0.9,
            tree_method="hist"
        )),
        "lightgbm": build_pipeline(LGBMClassifier(
            n_estimators=4000,random_state=RANDOM_STATE
        )),
    }

    results = {}
    for name, pipeline in models.items():
        # Train, evaluate, and save each fitted model
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        # Use the standard probability threshold
        predictions = (probabilities >= 0.5).astype(int)
        report = classification_report(y_test, predictions, output_dict=True)
        weighted = report["weighted avg"]
        results[name] = {
            # Use weighted scores
            "accuracy": round(report["accuracy"] * 100, 2),
            "precision": round(weighted["precision"] * 100, 2),
            "recall": round(weighted["recall"] * 100, 2),
            "f1_score": round(weighted["f1-score"] * 100, 2),
            "roc_auc": round(roc_auc_score(y_test, probabilities) * 100, 2),
        }
        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")
    return results, models

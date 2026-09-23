import os
from datetime import datetime, timezone
import json

import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


class MongoRepository:
    """MongoDB adapter for predictions, metrics, and summaries"""

    def __init__(self) -> None:
        # Read connection settings from the environment
        uri = os.getenv("MONGODB_URI")
        name = os.getenv("MONGODB_DATABASE")
        if not uri or not name:
            raise RuntimeError(
                "MONGODB_URI and MONGODB_DATABASE must be set in .env"
            )
        self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self.database = self.client[name]

    def check_connection(self) -> bool:
        # Use a short ping so the dashboard does not hang
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    @staticmethod
    def _records(dataframe: pd.DataFrame) -> list[dict]:
        """Convert pandas values into MongoDB-safe records"""
        return json.loads(dataframe.to_json(orient="records", date_format="iso"))

    def save_dataframe(self, collection_name: str, dataframe: pd.DataFrame) -> int:
        """Replace a collection with the latest pipeline output."""
        collection = self.database[collection_name]
        # Keep one current snapshot instead of duplicate pipeline runs
        collection.delete_many({})
        records = self._records(dataframe)

        for start in range(0, len(records), 5000):
            # Batch inserts keep large datasets manageable
            collection.insert_many(records[start:start + 5000])
        return len(records)

    def save_predictions(self, predictions: pd.DataFrame) -> int:
        return self.save_dataframe("predictions", predictions)

    def save_metrics(self, metrics: dict) -> None:
        self.database.model_metrics.insert_one({
            "created_at": datetime.now(timezone.utc), "metrics": metrics,
        })

    def save_summary(self, summary: dict) -> None:
        self.database.dashboard_summaries.insert_one({
            "created_at": datetime.now(timezone.utc), **summary,
        })

    def latest_metrics(self) -> dict | None:
        document = self.database.model_metrics.find_one(
            {}, sort=[("created_at", -1)]
        )
        return document.get("metrics") if document else None

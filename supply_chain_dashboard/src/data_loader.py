from io import BytesIO

import boto3
import pandas as pd


def load_from_s3(bucket_name: str, object_key: str) -> pd.DataFrame:
    """Load the supply-chain CSV directly from Amazon S3"""
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    return pd.read_csv(BytesIO(response["Body"].read()), encoding="latin1")

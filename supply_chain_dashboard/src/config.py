from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
# Active dataset location in Amazon S3
S3_BUCKET = "supply-chain-dataset-bucket"
S3_OBJECT_KEY = "DataCoSupplyChainDataset.csv"
# Keep experiments reproducible
RANDOM_STATE = 42

# Create the model folder when the project starts
MODEL_DIR.mkdir(parents=True, exist_ok=True)

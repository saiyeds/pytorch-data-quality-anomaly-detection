
import pandas as pd
from config.pipeline_config import *
from src.config_loader import load_config
from src.feature_engineering import generate_features
from src.train import train_model
from src.inference import score_anomalies
from src.powerbi_schema import to_powerbi

config = load_config("config/pipeline_config.yaml")

df = pd.DataFrame({
    "amount": [100, 200, 300, 400, 5000],
    "quantity": [1, 2, 3, 4, 50],
    "price": [10, 20, 30, 40, 500]
})

features = generate_features(df)
model = train_model(features, config)
errors, flags, threshold = score_anomalies(
    model, features, config["anomaly"]["threshold_percentile"]
)

print(to_powerbi(errors, flags))

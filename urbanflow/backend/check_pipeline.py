import joblib
import pandas as pd

pipe = joblib.load("data/lgbm_taxi_pipeline.pkl")

seed = pd.read_csv("data/historical_seed_data.csv")

features = set(pipe["feature_columns"])
seed_cols = set(seed.columns)

print("Missing from seed:")
print(features - seed_cols)

print("\nExtra in seed:")
print(seed_cols - features)
import pandas as pd
from datetime import datetime, timedelta
import os

print("Loading raw CSV data...")
df = pd.read_csv("data/raw/creditcard.csv")

print("Formatting for Feast Feature Store...")
# 1. Create a unique Entity ID (Transaction ID)
df['transaction_id'] = range(1, len(df) + 1)

# 2. Create a real Datetime Timestamp
# We will pretend the data started exactly 2 days ago from right now
start_time = datetime.now() - timedelta(days=2)
df['event_timestamp'] = df['Time'].apply(lambda x: start_time + timedelta(seconds=x))

# 3. Create the Feast repository folders
os.makedirs("feature_repo/data", exist_ok=True)

# 4. Save to Parquet (Feast's required offline format)
df.to_parquet("feature_repo/data/creditcard_features.parquet")
print("✅ Data formatted and saved as Parquet!")
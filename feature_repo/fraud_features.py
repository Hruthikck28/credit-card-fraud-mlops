from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32

# 1. Point Feast to our Parquet file
fraud_data = FileSource(
    name="creditcard_parquet",
    path="data/creditcard_features.parquet",
    timestamp_field="event_timestamp",
)

# 2. Define the primary key (Updated for your Feast version)
transaction = Entity(
    name="transaction_id",
    join_keys=["transaction_id"], # Added to be safe with newer Feast versions
    value_type=ValueType.INT64,   # Changed to use ValueType enum
    description="Unique identifier for the transaction",
)

# 3. Tell Feast which columns we care about
fraud_view = FeatureView(
    name="fraud_features",
    entities=[transaction],
    ttl=timedelta(days=365),
    schema=[
        Field(name="Amount", dtype=Float32),
        Field(name="V1", dtype=Float32),
        Field(name="V2", dtype=Float32),
        Field(name="V3", dtype=Float32),
        Field(name="V4", dtype=Float32),
        Field(name="V5", dtype=Float32),
    ],
    online=True,
    source=fraud_data,
)
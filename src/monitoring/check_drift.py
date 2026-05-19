import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

print("Loading reference training data...")
# 1. Read our original, clean data
reference_data = pd.read_parquet("feature_repo/data/creditcard_features.parquet")
# We just need the features the model uses
features = ['Amount', 'V1', 'V2', 'V3', 'V4', 'V5']
reference_data = reference_data[features].head(5000)

print("Simulating live production data with synthetic drift...")
# 2. Create a fake "new week" of data where user behavior changed drastically
current_data = reference_data.copy()
current_data['Amount'] = current_data['Amount'] * 5.0  # Massive spike in transaction sizes
current_data['V2'] = current_data['V2'] - 3.0          # Shift a PCA feature

print("Generating Evidently Data Drift report...")
# 3. Setup the Evidently report
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=reference_data, current_data=current_data)

# 4. Save the beautiful HTML dashboard
output_file = "drift_report.html"
drift_report.save_html(output_file)

print(f"✅ Drift check complete! Report saved to {output_file}")
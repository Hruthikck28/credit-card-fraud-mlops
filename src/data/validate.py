import great_expectations as ge

def validate_fraud_data(file_path):
    print(f"Loading data from {file_path}...")
    df = ge.read_csv(file_path)

    print("Running data quality gates...")
    # Rule 1: The target variable 'Class' must be strictly 0 (legit) or 1 (fraud)
    df.expect_column_values_to_be_in_set("Class", [0, 1])

    # Rule 2: You can't have a transaction with a negative amount
    df.expect_column_values_to_be_between("Amount", min_value=0)

    # Rule 3: The 'Time' column (seconds elapsed) shouldn't be negative
    df.expect_column_values_to_be_between("Time", min_value=0)

    # Run the validation
    result = df.validate()

    if result["success"]:
        print("✅ Data validation passed! Safe to proceed to feature engineering.")
    else:
        print("❌ Data validation failed! Bad data detected.")

if __name__ == "__main__":
    validate_fraud_data("data/raw/creditcard.csv")
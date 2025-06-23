import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile
import os

# Define file paths
zip_path = 'data/creditcard.csv.zip'
extract_path = 'data/'
csv_path = os.path.join(extract_path, 'creditcard.csv')

# Create data folder if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract dataset
if os.path.exists(zip_path) and zipfile.is_zipfile(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Dataset extracted successfully.")
else:
    raise FileNotFoundError("⚠️ creditcard.csv.zip not found or invalid.")

# Load dataset
df = pd.read_csv(csv_path)
print(f"✅ Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Sanity check
if 'Class' not in df.columns or 'Amount' not in df.columns:
    raise ValueError("❌ Required columns missing in dataset.")

# Stratified split
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['Class'], random_state=42)

# Save training data
train_df.to_csv(os.path.join(extract_path, "train_data.csv"), index=False)

# Select relevant columns for test cases
feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
test_df = test_df[feature_cols]  # Clean column order

# Save test cases
test_df.to_csv(os.path.join(extract_path, "test_cases.csv"), index=False)

# Summary logs
print("✅ train_data.csv and test_cases.csv saved.")
print(f"🔍 Class Distribution:\n{df['Class'].value_counts().to_dict()}")

import pandas as pd

# Load full training data
df = pd.read_csv("data/train_data.csv")

# Split into legit and fraud
fraud_df = df[df["Class"] == 1]          # Keep all fraud cases
legit_df = df[df["Class"] == 0]          # Will reduce this

# Downsample legit transactions to reduce file size
legit_sample = legit_df.sample(n=60000, random_state=42)

# Combine and shuffle
reduced_df = pd.concat([fraud_df, legit_sample]).sample(frac=1, random_state=42)

# Save reduced version
reduced_df.to_csv("data/train_data_small.csv", index=False)

print(f"✅ Saved reduced dataset with shape: {reduced_df.shape}")
print(f"🔁 Fraud count: {len(fraud_df)}, Legit count: {len(legit_sample)}")

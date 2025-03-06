import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_customer_churn.csv")

# Print feature names (excluding target 'churn')
print("🔹 Model expects the following features:")
print(df.drop(columns=['churn']).columns.tolist())
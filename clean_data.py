import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/customer_churn.csv")

# Standardize column names (convert to lowercase, replace spaces with underscores)
df.columns = df.columns.str.strip().str.lower()

# 🔹 Drop unnecessary columns
df.drop(columns=['customerid'], inplace=True)

# 🔹 Convert 'TotalCharges' to numeric (handling empty strings first)
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')

# 🔹 Fill missing values in 'TotalCharges' with the median
df['totalcharges'].fillna(df['totalcharges'].median(), inplace=True)

# 🔹 Encode categorical features
categorical_columns = ['gender', 'partner', 'dependents', 'phoneservice', 'multiplelines',
                       'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection',
                       'techsupport', 'streamingtv', 'streamingmovies', 'contract',
                       'paperlessbilling', 'paymentmethod', 'churn']

for col in categorical_columns:
    df[col] = df[col].astype('category').cat.codes  # Convert categorical values to numeric codes

# 🔹 Save cleaned dataset
df.to_csv("data/cleaned_customer_churn.csv", index=False)
print("✅ Data Cleaning Completed! Cleaned dataset saved as 'cleaned_customer_churn.csv'.")

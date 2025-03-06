import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_customer_churn.csv")

# 🔹 Display dataset information
print("\n🔹 First 5 rows of the dataset:")
print(df.head())

print("\n🔹 Dataset Info:")
print(df.info())

print("\n🔹 Summary Statistics:")
print(df.describe())

# 🔹 Check for missing values
missing_values = df.isnull().sum()
print("\n🔹 Missing Values:\n", missing_values[missing_values > 0])

# 🔸 1. Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df["churn"], palette="coolwarm")
plt.title("Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# 🔸 2. Contract Type vs. Churn Rate
plt.figure(figsize=(8, 5))
sns.barplot(x=df["contract"], y=df["churn"], palette="viridis")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type (Encoded)")
plt.ylabel("Churn Rate")
plt.show()

# 🔸 3. Monthly Charges Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["monthlycharges"], kde=True, bins=30)
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Count")
plt.show()

# 🔸 4. Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

print("\n✅ EDA Completed. Check visualizations for insights!")

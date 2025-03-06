import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🔹 Load cleaned dataset
df = pd.read_csv("data/cleaned_customer_churn.csv")

# 🔹 Separate features (X) and target variable (y)
X = df.drop(columns=['churn'])  # Features
y = df['churn']  # Target variable

# 🔹 Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Standardize numerical features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Store models in a dictionary
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# 🔹 Train and evaluate models
for name, model in models.items():
    print(f"\n🔹 Training {name}...\n")
    
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)  # Use scaled data
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)  # Use original data for tree-based models
        y_pred = model.predict(X_test)

    # Evaluate Model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ {name} Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, f"models/{name.lower().replace(' ', '_')}_model.pkl")

# Save the scaler (for models that require it)
joblib.dump(scaler, "models/scaler.pkl")

print("\n🎉 Model Training Completed! Trained models are saved in the 'models' folder.")

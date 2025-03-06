import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# 🔹 Ensure the "models" directory exists
if not os.path.exists("models"):
    os.makedirs("models")

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

# 🔹 Define hyperparameter grids
param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 10]
    }
}

# 🔹 Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# 🔹 Perform hyperparameter tuning
best_models = {}

for name, model in models.items():
    print(f"\n🔍 Tuning {name}...")
    
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    
    if name == "Logistic Regression":
        grid_search.fit(X_train_scaled, y_train)  # Use scaled data
    else:
        grid_search.fit(X_train, y_train)  # Use original data for tree-based models
    
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    
    print(f"✅ Best parameters for {name}: {grid_search.best_params_}")

    # Evaluate best model
    if name == "Logistic Regression":
        y_pred = best_model.predict(X_test_scaled)
    else:
        y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 {name} Accuracy after tuning: {accuracy:.4f}")
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save the best model
    joblib.dump(best_model, f"models/{name.lower().replace(' ', '_')}_best_model.pkl")

# Save the scaler (for models that require it)
joblib.dump(scaler, "models/scaler.pkl")

print("\n🎉 Hyperparameter Tuning Completed! Best models are saved in the 'models' folder.")

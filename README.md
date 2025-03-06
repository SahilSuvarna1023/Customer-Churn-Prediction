📊 Customer Churn Prediction

📌 Project Overview

Customer churn prediction helps businesses retain customers by identifying those likely to leave. This project uses machine learning to predict customer churn and deploys the model using a Flask API.

📂 Project Structure

customer_churn_project/
│── clean_data.py       # Data cleaning script
│── eda.py              # Exploratory Data Analysis
│── train_model.py      # Train ML models
│── tune_model.py       # Hyperparameter tuning
│── churn_api.py        # Flask API for predictions
│── data/
│   ├── customer_churn.csv  # Raw dataset
│   ├── cleaned_customer_churn.csv  # Cleaned dataset
│── models/
│   ├── random_forest_best_model.pkl  # Best trained model
│   ├── scaler.pkl  # Scaler for Logistic Regression
│── requirements.txt  # Dependencies
│── Procfile          # For Render deployment
│── README.md         # Project documentation

📦 Installed Libraries

pip install pandas numpy matplotlib seaborn scikit-learn xgboost flask joblib

🛠 Steps to Run the Project

1️⃣ Data Cleaning

python clean_data.py

2️⃣ Exploratory Data Analysis (EDA)

python eda.py

3️⃣ Train Machine Learning Models

python train_model.py

4️⃣ Tune Hyperparameters

python tune_model.py

5️⃣ Run Flask API

python churn_api.py

✅ API URL: http://127.0.0.1:5000

6️⃣ Test API using cURL

curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [1, 0, 1, 0, 12, 1, 0, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 45.3, 540.5]}'

🏆 Model Performance

Accuracy: ~85%

Precision, Recall, F1-score: See terminal output

Confusion Matrix:



🔥 Future Improvements

Further model optimization with deep learning

Deploying API on AWS Lambda for scalability

Creating a Streamlit UI for easier interaction

🤝 Contributing

Feel free to fork this repo and submit pull requests! Contributions are welcome. 😊

📜 License

This project is licensed under the MIT License.


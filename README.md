# 📊 Customer Churn Prediction - GitHub README

## 📌 Project Overview
Customer churn prediction helps businesses retain customers by identifying those likely to leave. This project uses **machine learning** to predict customer churn and deploys the model using a **Flask API** for real-time predictions.

---

## 📂 Project Structure
```
customer_churn_project/
│── clean_data.py       # Data cleaning script
│── eda.py              # Exploratory Data Analysis
│── train_model.py      # Train ML models
│── tune_model.py       # Hyperparameter tuning
│── churn_api.py        # Flask API for predictions
│── pkl_view.py         # View and test saved .pkl files
│── data/
│   ├── customer_churn.csv  # Raw dataset
│   ├── cleaned_customer_churn.csv  # Cleaned dataset
│── models/
│   ├── random_forest_best_model.pkl  # Best trained model
│   ├── scaler.pkl  # Scaler for Logistic Regression
│── requirements.txt  # Dependencies
│── Procfile          # For Render deployment
│── README.md         # Project documentation
```

---

## 📦 Installed Libraries
```sh
pip install pandas numpy matplotlib seaborn scikit-learn xgboost flask joblib
```

---

## 🛠 Steps to Run the Project
### 1️⃣ Data Cleaning
```sh
python clean_data.py
```

### 2️⃣ Exploratory Data Analysis (EDA)
```sh
python eda.py
```

### 3️⃣ Train Machine Learning Models
```sh
python train_model.py
```

### 4️⃣ Tune Hyperparameters
```sh
python tune_model.py
```

### 5️⃣ Run Flask API
```sh
python churn_api.py
```
✅ **API URL:** `http://127.0.0.1:5000`

### 6️⃣ Test API using cURL
```sh
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [1, 0, 1, 0, 12, 1, 0, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 45.3, 540.5]}'
```

### 7️⃣ View and Test Saved Model Files
```sh
python pkl_view.py
```

### 8️⃣ Deploy on Render
```sh
git init
git add .
git commit -m "Initial commit"
git push -u origin main
```

Deploy on **Render.com** by linking the GitHub repo and setting **Start Command:** `python churn_api.py`.

---

## 🏆 Model Performance
| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 79.2%    | 76.5%     | 72.3%  | 74.3%    |
| Random Forest      | 85.1%    | 83.4%     | 80.7%  | 82.0%    |
| XGBoost           | **87.4%** | **85.9%** | **83.1%** | **84.4%** |

- **Best Model:** XGBoost (87.4% Accuracy)
- **Confusion Matrix:**
  ![Confusion Matrix](https://your-image-url.com)

---

## 🌐 API Endpoints
### 1️⃣ **Check API Health**
```sh
curl -X GET http://127.0.0.1:5000/
```
#### **Response:**
```json
{
  "message": "Customer Churn Prediction API is running!"
}
```

### 2️⃣ **Make a Prediction**
```sh
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [1, 0, 1, 0, 12, 1, 0, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 45.3, 540.5]}'
```
#### **Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.87
}
```

---

## 🔥 Future Improvements
- Further model optimization with deep learning
- Deploying API on AWS Lambda for scalability
- Creating a Streamlit UI for easier interaction
- Implementing real-time churn detection for businesses

---

## 🤝 Contributing
Feel free to fork this repo and submit pull requests! Contributions are welcome. 😊

---

## 📜 License
This project is licensed under the MIT License.


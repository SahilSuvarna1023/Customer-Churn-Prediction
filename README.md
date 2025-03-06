# 📊 Customer Churn Prediction

## 📌 Project Overview
Customer churn prediction is essential for businesses to retain customers and reduce revenue loss. This project uses **machine learning** to predict whether a customer will churn based on their behavior and demographics.

---

## 📂 Project Structure
```
customer_churn_project/
│── churn_model.py       # Main script for training & evaluation
│── churn_app.py         # Flask API for predictions
│── data/
│   ├── customer_churn.csv  # Dataset
│── models/
│   ├── churn_model.pkl     # Saved trained model
│── requirements.txt        # Dependencies
│── README.md               # Project documentation
```

---

## 📊 Dataset Information
We use a customer churn dataset containing various features:
- **Customer ID**: Unique identifier
- **Gender**: Male/Female
- **Subscription Type**: Monthly, Yearly
- **Monthly Charges**: Payment made per month
- **Total Charges**: Cumulative amount paid
- **Contract Type**: Month-to-month, one-year, two-year
- **Churn**: **Target Variable** (1 = Churned, 0 = Not Churned)

**Dataset Source:** [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🛠 Installation
Clone the repository and install dependencies:
```sh
git clone https://github.com/yourusername/customer_churn_project.git
cd customer_churn_project
pip install -r requirements.txt
```

---

## 🏗 Steps to Run the Model
### 1️⃣ Train and Evaluate Model
Run the script to train and evaluate the model:
```sh
python churn_model.py
```
**Outputs:**
- Model accuracy & classification report
- Confusion matrix visualization
- Saved model as `churn_model.pkl`

### 2️⃣ Start the API Server (Optional)
To deploy the model using Flask, run:
```sh
python churn_app.py
```
This will start a server at **http://127.0.0.1:5000**

### 3️⃣ Make a Prediction
Send a POST request with customer features:
```sh
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [25, 0, 50.0, 1]}'
```
**Response:**
```json
{"churn_prediction": 1}
```

---

## 🏆 Model Performance
- **Accuracy:** ~85%
- **Precision, Recall, F1-score:** See terminal output
- **Confusion Matrix:**
  
  ![Confusion Matrix](https://your-image-url.com)

---

## 🔥 Future Improvements
- Hyperparameter tuning using GridSearchCV
- Feature engineering for better accuracy
- Deploying the model on **AWS or Heroku**

---

## 🤝 Contributing
Feel free to fork this repo and submit pull requests! Contributions are welcome. 😊

---

## 📜 License
This project is licensed under the MIT License.


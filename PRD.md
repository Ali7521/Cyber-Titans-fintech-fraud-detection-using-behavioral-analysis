Below is a **complete PRD (Product Requirement Document) in Markdown** format that you can directly use with **Claude Opus 4.6** to guide AI for building your project.

Since you mentioned:

* Dataset is already attached in the same directory
* Project = **Fraud Detection System using Behavioral Analysis**
* You’ll use Claude to generate code
* Likely using Streamlit (based on your previous hackathon context)

This PRD is written in a way that Claude can clearly understand architecture, constraints, and output expectations.

---

# 📌 Product Requirement Document (PRD)

# 🛡️ Fraud Detection System using Behavioral Analysis

---

## 1. 📖 Project Overview

### Product Name:

Behavioral Fraud Detection System

### Objective:

Build a machine learning-based fraud detection system that analyzes user behavioral patterns from the provided dataset and predicts fraudulent transactions in real-time.

The dataset is already placed in the same project directory.

The system must:

* Load dataset from local directory
* Perform preprocessing
* Extract behavioral features
* Train ML models
* Evaluate performance
* Provide a Streamlit-based UI for predictions

---

## 2. 🎯 Goals

* Detect fraudulent transactions with high accuracy
* Focus on behavioral anomaly detection
* Minimize false positives
* Provide interactive dashboard for monitoring
* Allow real-time transaction input & prediction

---

## 3. 📂 Dataset Details

* Dataset file is located in the same project directory.
* Format: CSV
* Contains transaction-level data.
* Includes a target column (e.g., `Class`, `isFraud`, or similar).

AI should:

1. Automatically detect target column.
2. Perform exploratory data analysis (EDA).
3. Identify behavioral features such as:

   * Transaction frequency
   * Transaction amount patterns
   * Time-based patterns
   * Device or location changes
   * Velocity features

---

## 4. 🧠 Core Concept: Behavioral Analysis

The system must focus on:

### Behavioral Signals:

* Sudden spike in transaction amount
* Multiple transactions in short time
* Change in geolocation/device
* Unusual time-of-day activity
* Deviation from user's historical average

AI must engineer features such as:

* Rolling averages
* Standard deviation of user transactions
* Time difference between transactions
* Frequency per hour/day
* Z-score based anomaly detection

---

## 5. 🏗️ System Architecture

### Step 1: Data Loading

* Use pandas to load CSV from current directory.
* Validate schema.
* Handle missing values.

### Step 2: Preprocessing

* Encode categorical variables
* Scale numerical features
* Handle class imbalance using:

  * SMOTE or
  * Class weighting

### Step 3: Feature Engineering

* Generate behavioral features
* Normalize data

### Step 4: Model Training

Implement and compare:

* Logistic Regression
* Random Forest
* XGBoost
* Isolation Forest (for anomaly detection)

Select best model based on:

* ROC-AUC
* Precision
* Recall
* F1-score

### Step 5: Model Evaluation

Generate:

* Confusion matrix
* ROC curve
* Feature importance
* Precision-Recall curve

### Step 6: Model Saving

* Save best model using joblib or pickle

---

## 6. 🖥️ Frontend (Streamlit App)

The app must contain:

### Page 1: Dashboard

* Total transactions
* Fraud rate
* Model accuracy
* Visualizations:

  * Fraud vs Non-Fraud
  * Transaction amount distribution
  * Time-based fraud pattern

### Page 2: Real-Time Prediction

* Input fields for transaction details
* Predict fraud probability
* Show risk score
* Show explanation (SHAP or feature importance)

### Page 3: Model Insights

* Feature importance chart
* ROC curve
* Confusion matrix

---

## 7. 📊 Success Metrics

| Metric               | Target        |
| -------------------- | ------------- |
| Accuracy             | > 95%         |
| Recall (Fraud class) | High priority |
| Precision            | Balanced      |
| False Positive Rate  | Low           |

---

## 8. 🔐 Non-Functional Requirements

* Clean modular code
* Reusable functions
* Separate files:

  * data_processing.py
  * feature_engineering.py
  * model_training.py
  * app.py
* Well-commented code
* Production-ready structure

---

## 9. 📁 Suggested Project Structure

```
fraud-detection/
│
├── dataset.csv
├── app.py
├── data_processing.py
├── feature_engineering.py
├── model_training.py
├── model.pkl
├── requirements.txt
└── README.md
```

---

## 10. 🧾 Dependencies

Include in requirements.txt:

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
streamlit
joblib
shap
```

---

## 11. 🤖 Instructions for AI (Claude Opus 4.6)

When generating code:

1. Assume dataset is already available locally.
2. Automatically detect column names.
3. Write modular production-level code.
4. Explain each major step briefly.
5. Handle errors gracefully.
6. Optimize for imbalanced classification.
7. Include model comparison table.
8. Generate complete working Streamlit app.

---

## 12. 🧪 Testing Requirements

* Test with unseen split (20% test set)
* Use cross-validation
* Show classification report
* Validate model overfitting

---

## 13. 🚀 Future Improvements

* Real-time streaming integration
* API deployment using FastAPI
* Model monitoring
* Drift detection
* Online learning

---

# ✅ Expected Final Output

* Trained fraud detection model
* Behavioral feature engineering
* Interactive Streamlit dashboard
* Model evaluation visuals
* Production-ready folder structure

---

If you want, I can now:

* 🔥 Make a **more advanced enterprise-level PRD**
* 💡 Or convert this into a **Claude-ready system prompt**
* 🚀 Or give you exact prompt you should paste into Claude Opus 4.6**

Tell me which level you want — hackathon level or production startup level?

# AI-Powered Bank Fraud Detection System

## Overview

This project is an AI-powered fraud detection system designed to identify potentially fraudulent bank transactions using machine learning. The model analyzes transaction details and predicts whether a transaction is legitimate or fraudulent, helping financial institutions reduce fraud losses and improve customer security.

---

## Features

* Fraud detection using machine learning
* Data preprocessing and feature engineering
* Automatic handling of categorical and numerical features
* Probability-based fraud prediction
* Adjustable decision threshold for different business requirements
* Model persistence for deployment using Joblib

---

## Dataset

The project uses the **Bank Account Fraud Dataset (NeurIPS 2022)**.

The dataset contains transaction and customer-related information, including:

* Customer income
* Credit risk score
* Payment type
* Housing status
* Employment status
* Device operating system
* Transaction velocity
* Session information
* Address history
* Banking history
* Fraud label (`fraud_bool`)

For faster experimentation, a reduced dataset containing:

* **40,000 legitimate transactions**
* **11,029 fraudulent transactions**

was used during model development.

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Joblib

---

## Project Structure

```text
.
├── Base_sample_40k.csv
├── train_model.py
├── lightgbm_fraud_model.pkl
├── README.md
```

---

## Data Preprocessing

The preprocessing pipeline performs the following steps:

* Handle missing values
* Replace invalid values (-1) with missing values
* Standardize numerical features
* One-Hot Encode categorical features
* Split the dataset into training and testing sets

The preprocessing pipeline is saved together with the trained model.

---

## Model

The final model is based on **LightGBM** with class weighting to reduce the impact of class imbalance.

Main parameters include:

* Learning Rate: 0.05
* Number of Trees: 250
* Maximum Depth: 6
* Class Weighting using `scale_pos_weight`

---

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Precision-Recall AUC (PR-AUC)
* Confusion Matrix

A custom probability threshold is used to balance precision and recall according to the application's requirements.

---

## Running the Project

### Install dependencies

```bash
pip install pandas numpy scikit-learn lightgbm joblib
```

### Train the model

```bash
python train_model.py
```

---

## Saving the Model

The trained model is saved as:

```text
lightgbm_fraud_model.pkl
```

This file contains both the preprocessing pipeline and the trained LightGBM model.

---

## Using the Saved Model

```python
import joblib

model = joblib.load("lightgbm_fraud_model.pkl")

prediction = model.predict(new_data)
probability = model.predict_proba(new_data)
```

---

## Future Improvements

* Hyperparameter optimization
* CatBoost and XGBoost comparison
* Ensemble learning
* Explainable AI using SHAP
* Real-time fraud detection API
* Dashboard for fraud monitoring

---

## Authors

Developed as an AI-powered fraud detection project using machine learning techniques for secure banking transaction analysis.

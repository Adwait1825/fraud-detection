# Credit Card Fraud Detection

A machine learning project comparing a logistic regression baseline against a feedforward neural network for detecting fraudulent credit card transactions.

## Project Overview
This project addresses binary classification on a highly imbalanced dataset (~0.17% fraud cases). It trains and evaluates two models:
- **Baseline:** Logistic Regression (scikit-learn)
-**Main model:** Feedforward Neural Network (PyTorch) 


## Dataset
[Credit Card Fraud Detection dataset](https://www.kaggle.com/dataset/mlg-vlb/creditcardfraud) from the Machine Learning Group at ULB - 284,807 transactions, 492 landed as fraud. Features V1-V28 are PCA-transformed for confidentiality; 'Time' and 'Amount' are the only raw features.

Not include in this repository due to size - download it from Kaggle and place 'creditcard.csv' in the 'data/' folder before running.

## How to Run
1. Clone repository
2. Create and activate a virtual environment:
3. Install dependencies
4. Download the dataset  from Kaggle and place 'creditcard.csv' in the 'data/' folder
5. Run the full pipeline:

## Results 
| Metric | Logistic Regression | Neural Network |
|---|---|---|
| ROC-AUC | ~0.972 | ~0.977 |
| Fraud recall | ~0.972 | ~0.90-0.95 |
| Fraud precision | ~0.06 | ~0.04-0.05 |

Full evaluation details and discussion are in the project report.

## Author 
Adwait Khaladkar
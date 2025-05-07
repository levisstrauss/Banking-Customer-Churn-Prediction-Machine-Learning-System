
# Customer Churn Prediction System

## Project Overview
This repository contains a production-ready machine learning system for predicting customer churn in the banking 
industry. The project applies software engineering best practices to create a maintainable, well-tested, and 
well-documented ML solution.

## Problem Statement
Customer churn (when customers stop using a company's services) significantly impacts business revenue and growth. 
Predicting which customers are likely to churn enables proactive retention strategies. This project leverages machine 
learning to identify patterns that indicate increased churn probability.

## Approach

The project implements a complete ML pipeline:
1. Data Ingestion & Exploration: Importing banking customer data and performing comprehensive exploratory data analysis
2. Feature Engineering: Transforming categorical variables and preparing data for modeling
3. Model Training: Implementing and optimizing Random Forest and Logistic Regression models
4. Evaluation: Assessing model performance with classification metrics and visualizations
5. Feature Importance Analysis: Identifying key factors influencing customer churn predictions

## Project Structure

```bash
customer_churn_prediction/
├── data/                 # Contains the dataset
├── images/               # Visualization outputs
│   ├── eda/              # Exploratory data analysis plots
│   └── results/          # Model performance visualizations
├── logs/                 # Execution and test logs
├── models/               # Saved model files (.pkl)
├── churn_library.py      # Main ML functions library
├── test_churn_script_logging_and_tests.py  # Unit tests
├── pytest.ini           # Pytest configuration
└── requirements.txt     # Project dependencies
```
## Key Features
- Production-Ready Code: Follows PEP8 style guidelines and scores >8.0 on pylint
- Comprehensive Documentation: Detailed docstrings and comments
- Error Handling: Robust validation and error reporting
- Testing Framework: Complete test suite using pytest
- Logging System: Detailed execution logs for monitoring and debugging
- Visualization Suite: Insightful charts for both EDA and model performance
- 
## Model Performance
The Random Forest classifier outperforms the Logistic Regression model:

- Higher ROC-AUC score and superior precision-recall balance
- Effective identification of churn risk factors
- Strong predictive performance on unseen data

The confusion matrix analysis shows that while the model has strong overall accuracy, there are some false negatives 
that would need special attention in a production deployment.

## Feature Importance
Analysis reveals that the most significant predictors of customer churn include:

- Total transaction count
- Total transaction amount
- Customer age
- Credit limit utilization

These insights provide actionable intelligence for customer retention strategies.

## Installation & Usage

Prerequisites
- Python 3.8
- Libraries listed in requirements.txt

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Udacity Machine Learning DevOps Engineer Nanodegree program

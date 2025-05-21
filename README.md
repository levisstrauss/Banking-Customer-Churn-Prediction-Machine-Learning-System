
# 📊 Banking Customer Churn Prediction

## 🔍 Project Overview

This repository contains a machine learning project focused on predicting customer churn in the banking industry. Created as a portfolio project to 
demonstrate data science and ML engineering skills, it applies software engineering best practices to create a well-tested and well-documented prediction system.

## Why Customer Churn Matters

Customer churn (when customers stop using a company's services) significantly impacts business revenue and growth. In banking specifically:

- Acquiring new customers costs 5-25x more than retaining existing ones
- Even small improvements in retention rates can have significant financial implications
- Identifying at-risk customers before they leave enables proactive intervention

This project explores how machine learning can identify patterns that indicate increased churn probability, using best software engineering best 
practices and clean production python code, providing insights that could potentially inform customer retention strategies.

## 💡 Solution Approach

This project implements a complete ML pipeline demonstrating best practices in data science:

## Data Exploration & Processing

- Comprehensive EDA: Thorough exploration of banking customer data with visualization
- Data Cleaning: Handling of missing values and outliers
- Feature Engineering: Creating meaningful predictors from raw banking data
- Data Transformation: Preparing categorical and numerical variables for modeling

## Model Development

- Multiple Algorithms: Implementation of Random Forest and Logistic Regression
- Hyperparameter Optimization: Grid search for model tuning
- Performance Evaluation: Comprehensive metrics including ROC-AUC, precision, recall, and F1
- Feature Importance Analysis: Identifying key factors that predict customer churn

## Software Engineering Best Practices

- Modular Design: Well-structured code with separation of concerns
- Documentation: Comprehensive docstrings and comments
- Testing: Complete test suite using pytest
- Logging: Detailed execution logs for process monitoring
- Code Quality: Adherence to PEP8 style guidelines

## 🚀 Project Implementation

## Code Quality Example  

```python

def perform_feature_engineering(df, response=None):
    """
    Engineer features for machine learning model from preprocessed data.
    
    Args:
        df (pandas.DataFrame): Preprocessed data
        response (str, optional): Target variable name. Defaults to 'Churn'.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test - split and prepared modeling datasets
    
    Raises:
        ValueError: If critical features are missing
        TypeError: If df is not a pandas DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        logging.error("Input is not a pandas DataFrame")
        raise TypeError("Input must be a pandas DataFrame")
        
    try:
        # Validate expected columns present
        expected_features = ['Customer_Age', 'Dependent_count', 'Total_Trans_Ct']
        missing_cols = [col for col in expected_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing critical features: {missing_cols}")

        # Feature Engineering implementation
        y = df[response] if response else df['Churn']
        X = pd.DataFrame()
        
        # Category Encodings
        cat_columns = ['Gender', 'Education_Level', 'Marital_Status', 
                       'Income_Category', 'Card_Category']
        X = pd.get_dummies(df, columns=cat_columns, drop_first=True)
        
        # Remove target from features
        if response in X.columns:
            X = X.drop([response], axis=1)
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        logging.info("Feature engineering successful: %s features created", X.shape[1])
        return X_train, X_test, y_train, y_test
        
    except Exception as err:
        logging.error("Feature engineering failed: %s", str(err))
        raise
```

## Testing Approach
The project includes comprehensive testing to ensure reliability:

```python
def test_perform_eda_creates_expected_plots():
    """
    Test that perform_eda function creates all expected exploratory plots.
    """
    # Setup
    df = import_data("./data/bank_data.csv")
    expected_files = [
        './images/eda/customer_age_distribution.png',
        './images/eda/marital_status_distribution.png',
        './images/eda/transaction_heatmap.png',
        './images/eda/churn_distribution.png'
    ]
    
    # Remove any existing files for clean test
    for file in expected_files:
        if os.path.exists(file):
            os.remove(file)
    
    # Execute
    perform_eda(df)
    
    # Assert
    for file in expected_files:
        assert os.path.exists(file), f"EDA failed to create {file}"
        assert os.path.getsize(file) > 0, f"EDA created empty file: {file}"
```
## 📊 Model Performance

This project explores two machine learning models for churn prediction:

## Confusion Matrix Analysis
The confusion matrix shows the Random Forest model's prediction results:

- True Positives: Correctly identified customers likely to churn
- False Positives: Customers incorrectly flagged as likely to churn
- False Negatives: At-risk customers that the model failed to identify
- True Negatives: Correctly identified stable customers

## Feature Importance Analysis

The analysis reveals several important predictors of customer churn:

1. Total_Trans_Ct: Transaction frequency is the strongest predictor
2. Total_Trans_Amt: Total spending volume is highly relevant
3. Customer_Age: Customer tenure affects churn likelihood
4. Credit_Limit: Available credit shows relationship with retention

These insights align with common banking industry knowledge that customer engagement (measured through transaction activity) is strongly correlated with retention.

## 🧪 Key Learnings

This project demonstrates several important aspects of applied machine learning:

1. Imbalanced Classification: Techniques for handling the typical class imbalance in churn prediction
2. Feature Engineering: Creating meaningful predictors from banking transaction data
3. Model Comparison: Evaluating tradeoffs between different algorithms
4. Software Engineering: Applying best practices to data science workflows

## 🚀 Getting Started

## Prerequisites

- Python 3.8+
- Libraries listed in requirements.txt

## Installation

```bash
# Clone repository
git clone https://github.com/levisstrauss/Banking-Customer-Churn-Prediction.git
cd Banking-Customer-Churn-Prediction

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## Running the Pipeline
```bash
# Run the main script
python churn_library.py

# Run tests
python -m pytest test_churn_script_logging_and_tests.py -v
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Udacity Machine Learning DevOps Engineer Nanodegree program for project inspiration
- The scikit-learn team for their excellent documentation and examples










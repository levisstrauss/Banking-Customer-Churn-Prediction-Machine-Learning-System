
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict Customer Churn

This module contains utility functions for identifying and predicting customer churn
in a banking dataset. It performs exploratory data analysis, feature engineering,
and trains machine learning models to predict customer churn.

Author: Zakaria Coulibaly
Date: May 7, 2025
"""

# Import libraries
import os
import logging
import joblib
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
# Configure seaborn
sns.set_theme(style="whitegrid")

# Configure logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('churn_library')

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth: str) -> pd.DataFrame:
    """
    Import dataset from a CSV file and prepare it for analysis.

    Args:
        pth: Path to the CSV file

    Returns:
        DataFrame containing the prepared dataset with encoded churn variable

    Raises:
        FileNotFoundError: If the file at the specified path does not exist
    """
    try:
        dataframe = pd.read_csv(pth, index_col=0)
        logger.info("Data imported successfully from %s", pth)

        # Encode Churn dependent variable: 0 = Did not churn; 1 = Churned
        dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        # Drop redundant Attrition_Flag variable (replaced by Churn response
        # variable)
        dataframe.drop('Attrition_Flag', axis=1, inplace=True)

        # Drop variable not relevant for the prediction model
        dataframe.drop('CLIENTNUM', axis=1, inplace=True)

        logger.info("Data preparation complete. Shape: %s", dataframe.shape)
        return dataframe
    except FileNotFoundError as err:
        logger.error("ERROR: File not found at %s", pth)
        raise err


def perform_eda(dataframe: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis on the dataset and save visualizations.

    Args:
        dataframe: Pandas DataFrame to analyze

    Returns:
        None
    """
    logger.info("Starting exploratory data analysis")

    # Create images/eda directory if it doesn't exist
    os.makedirs("./images/eda", exist_ok=True)

    # Analyze categorical features and plot distributions
    cat_columns = dataframe.select_dtypes(include='object').columns.tolist()
    for cat_column in cat_columns:
        plt.figure(figsize=(10, 5))
        (dataframe[cat_column]
         .value_counts('normalize')
         .plot(kind='bar',
               rot=45,
               title=f'{cat_column} - % Distribution')
         )
        plt.tight_layout()
        plt.savefig(os.path.join("./images/eda", f'{cat_column}.png'))
        plt.close()
        logger.info("Created distribution plot for %s", cat_column)

    # Analyze numeric features
    plt.figure(figsize=(10, 5))
    dataframe['Customer_Age'].plot(
        kind='hist',
        title='Distribution of Customer Age'
    )
    plt.tight_layout()
    plt.savefig(os.path.join("./images/eda", 'Customer_Age.png'))
    plt.close()
    logger.info("Created histogram for Customer_Age")

    plt.figure(figsize=(10, 5))
    # Show distributions of 'Total_Trans_Ct' with kernel density estimate
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Distribution of Total Transaction Count')
    plt.tight_layout()
    plt.savefig(os.path.join("./images/eda", 'Total_Trans_Ct.png'))
    plt.close()
    logger.info("Created histogram with KDE for Total_Trans_Ct")

    # Plot correlation matrix
    plt.figure(figsize=(20, 10))
    # Filter to only use numeric columns for correlation
    numeric_df = dataframe.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()

    plt.savefig(os.path.join("./images/eda", 'correlation_matrix.png'))
    plt.close()
    logger.info("Created correlation matrix heatmap")

    # Scatter plot to analyze relationship between transaction amount and count
    plt.figure(figsize=(15, 7))
    dataframe.plot(
        x='Total_Trans_Amt',
        y='Total_Trans_Ct',
        kind='scatter',
        title='Relationship: Transaction Amount vs. Count'
    )
    plt.tight_layout()
    plt.savefig(os.path.join("./images/eda", 'trans_amt_vs_count.png'))
    plt.close()
    logger.info("Created scatter plot for transaction relationships")

    logger.info("Exploratory data analysis complete")


def encoder_helper(
        dataframe: pd.DataFrame,
        category_lst: List[str],
        response: str = 'Churn'
) -> pd.DataFrame:
    """
    Encode categorical features using mean response value.

    For each category in each categorical column, replace the category with
    the mean response value for that category.

    Args:
        dataframe: Pandas DataFrame to encode
        category_lst: List of categorical columns to encode
        response: Name of response variable

    Returns:
        DataFrame with categorical features encoded

    Raises:
        KeyError: If any column in category_lst is not found in the dataframe
    """
    logger.info("Starting categorical feature encoding")

    # Create a copy of the dataframe to avoid modifying the original
    result_df = dataframe.copy()

    # Validate all categories exist in dataframe
    missing_cols = [
        col for col in category_lst if col not in result_df.columns]
    if missing_cols:
        err_msg = f"The following columns were not found: {missing_cols}"
        logger.error(err_msg)
        raise KeyError(err_msg)

    # Encode each category with the mean of the response variable
    for category in category_lst:
        logger.info("Encoding %s with mean %s values", category, response)
        # Calculate mean response for each category value, explicitly set
        # numeric_only=True
        category_groups = result_df.groupby(category)[response].mean()
        new_feature = f"{category}_{response}"
        # Map the values using the computed means
        result_df[new_feature] = result_df[category].map(category_groups)

    # Drop the original categorical features
    result_df.drop(category_lst, axis=1, inplace=True)
    logger.info("Categorical feature encoding complete")

    return result_df


def perform_feature_engineering(
        dataframe: pd.DataFrame,
        response: str = 'Churn'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Engineer features and split data into training and testing sets.

    Args:
        dataframe: Pandas DataFrame to process
        response: Name of the target variable

    Returns:
        Tuple containing:
            X_train: Features training data
            X_test: Features testing data
            y_train: Target training data
            y_test: Target testing data
    """
    logger.info("Starting feature engineering")

    # Collect categorical features to be encoded
    cat_columns = dataframe.select_dtypes(include='object').columns.tolist()
    logger.info(
        "Identified %d categorical columns for encoding",
        len(cat_columns))

    # Encode categorical features
    try:
        dataframe = encoder_helper(dataframe, cat_columns, response)
    except KeyError as err:
        logger.error("Feature engineering failed: %s", err)
        raise

    # Separate target variable and features
    y = dataframe[response]
    X = dataframe.drop(response, axis=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    logger.info(
        "Feature engineering complete. Training set size: %s, Testing set size: %s",
        X_train.shape,
        X_test.shape)

    return X_train, X_test, y_train, y_test


def plot_classification_report(
        model_name: str,
        y_train: pd.Series,
        y_test: pd.Series,
        y_train_preds: np.ndarray,
        y_test_preds: np.ndarray
) -> None:
    """
    Create and save classification report plots for training and testing results.

    Args:
        model_name: Name of the model
        y_train: Training response values
        y_test: Testing response values
        y_train_preds: Predicted training values
        y_test_preds: Predicted testing values

    Returns:
        None
    """
    # Create the directory if it doesn't exist
    os.makedirs("./images/results", exist_ok=True)

    # Create figure for classification report
    plt.figure(figsize=(8, 6))

    # Plot classification report for training data
    plt.text(
        0.01,
        1.25,
        f"{model_name} Train",
        {'fontsize': 12, 'fontweight': 'bold'},
        fontproperties='monospace'
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_train, y_train_preds)),
        {'fontsize': 10},
        fontproperties='monospace'
    )

    # Plot classification report for testing data
    plt.text(
        0.01,
        0.6,
        f"{model_name} Test",
        {'fontsize': 12, 'fontweight': 'bold'},
        fontproperties='monospace'
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_test, y_test_preds)),
        {'fontsize': 10},
        fontproperties='monospace'
    )

    plt.axis('off')

    # Save figure to image folder
    fig_name = f'classification_report_{model_name}.png'
    plt.savefig(
        os.path.join("./images/results", fig_name),
        bbox_inches='tight'
    )
    plt.close()

    logger.info("Classification report for %s saved successfully", model_name)


def classification_report_image(
        y_train: pd.Series,
        y_test: pd.Series,
        y_train_preds_lr: np.ndarray,
        y_train_preds_rf: np.ndarray,
        y_test_preds_lr: np.ndarray,
        y_test_preds_rf: np.ndarray
) -> None:
    """
    Create and save classification reports for both models.

    Args:
        y_train: Training response values
        y_test: Testing response values
        y_train_preds_lr: Training predictions from logistic regression
        y_train_preds_rf: Training predictions from random forest
        y_test_preds_lr: Testing predictions from logistic regression
        y_test_preds_rf: Testing predictions from random forest

    Returns:
        None
    """
    logger.info("Creating classification report images")

    # Create classification report for logistic regression
    plot_classification_report(
        'Logistic_Regression',
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr
    )

    # Create classification report for random forest
    plot_classification_report(
        'Random_Forest',
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf
    )

    logger.info("Classification report images created successfully")


def feature_importance_plot(
        model: object,
        X_data: pd.DataFrame,
        model_name: str,
        output_pth: str
) -> None:
    """
    Create and save feature importance plot.

    Args:
        model: Model object containing feature_importances_ attribute
        X_data: Features dataset
        model_name: Name of the model
        output_pth: Path to save the plot

    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(output_pth, exist_ok=True)

    # Calculate feature importances
    try:
        importances = model.feature_importances_
    except AttributeError:
        logger.error(
            "%s model does not have feature_importances_ attribute",
            model_name)
        return

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names to match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 8))

    # Create plot title and labels
    plt.title(f"Feature Importance for {model_name}", fontsize=18)
    plt.ylabel('Importance', fontsize=14)
    plt.xlabel('Features', fontsize=14)

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()

    # Save figure
    fig_name = f'feature_importance_{model_name}.png'
    plt.savefig(os.path.join(output_pth, fig_name), bbox_inches='tight')
    plt.close()

    logger.info(
        "Feature importance plot for %s saved successfully",
        model_name)


def plot_confusion_matrix(
        model: object,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series
) -> None:
    """
    Create and save confusion matrix visualization.

    Args:
        model: Trained model object
        model_name: Name of the model
        X_test: Testing features
        y_test: Testing response values

    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs("./images/results", exist_ok=True)

    # Define class names for the confusion matrix
    class_names = ['Not Churned', 'Churned']

    # Create figure
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Create and plot confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        ax=ax
    )

    # Hide grid lines
    ax.grid(False)

    # Add title
    plt.title(f'{model_name} Confusion Matrix', fontsize=16)
    plt.tight_layout()

    # Save figure
    plt.savefig(
        os.path.join("./images/results", f'{model_name}_Confusion_Matrix.png'),
        bbox_inches='tight'
    )
    plt.close()

    logger.info("Confusion matrix for %s saved successfully", model_name)


def train_models(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
) -> None:
    """
    Train, evaluate, and save models.

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training response values
        y_test: Testing response values

    Returns:
        None
    """
    logger.info("Starting model training process")

    # Create directories if they don't exist
    os.makedirs("./images/results", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # Initialize Random Forest model
    rfc = RandomForestClassifier(random_state=42)

    # Initialize Logistic Regression model
    lrc = LogisticRegression(
        solver='liblinear',
        max_iter=3000,
        random_state=42)

    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Set up grid search for Random Forest
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1)

    # Train Random Forest using GridSearch
    logger.info("Training Random Forest model with GridSearchCV")
    cv_rfc.fit(X_train, y_train)
    logger.info("Random Forest best parameters: %s", cv_rfc.best_params_)

    # Train Logistic Regression
    logger.info("Training Logistic Regression model")
    lrc.fit(X_train, y_train)

    # Get predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Generate classification reports
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf
    )

    # Plot ROC curves
    plt.figure(figsize=(15, 10))
    ax = plt.gca()

    # Plot Random Forest ROC curve
    rfc_disp = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        name="Random Forest",
        ax=ax,
        alpha=0.8
    )

    # Plot Logistic Regression ROC curve
    lrc_disp = RocCurveDisplay.from_estimator(
        lrc,
        X_test,
        y_test,
        name="Logistic Regression",
        ax=ax,
        alpha=0.8
    )

    # Add title and labels
    plt.title("ROC Curve Comparison", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save ROC curves
    plt.savefig(
        os.path.join("./images/results", 'ROC_curves.png'),
        bbox_inches='tight'
    )
    plt.close()
    logger.info("ROC curves saved successfully")

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    logger.info("Models saved successfully")

    # Create confusion matrices for both models
    for model, model_name in zip(
            [cv_rfc.best_estimator_, lrc],
            ['Random_Forest', 'Logistic_Regression']
    ):
        plot_confusion_matrix(model, model_name, X_test, y_test)

    # Create feature importance plot for Random Forest
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        'Random_Forest',
        "./images/results"
    )

    logger.info("Model training and evaluation complete")


if __name__ == "__main__":
    # Load and prepare data
    dataset = import_data("./data/bank_data.csv")
    logger.info("Dataset successfully loaded with shape: %s", dataset.shape)

    # Perform exploratory data analysis
    logger.info("Starting exploratory data analysis...")
    perform_eda(dataset)

    # Engineer features and split data
    logger.info("Starting feature engineering...")
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        dataset, response='Churn')

    # Train and evaluate models
    logger.info("Starting model training and evaluation...")
    train_models(X_train, X_test, y_train, y_test)

    logger.info("Process completed successfully")

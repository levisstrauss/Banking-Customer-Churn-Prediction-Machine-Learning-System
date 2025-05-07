"""
Unit test module for churn_library.py

This module contains unit tests for the functions in churn_library.py
using pytest framework.

Author: Zakaria Coulibaly
Date: May 7, 2025
"""


import os
import logging
import pytest
import pandas as pd
import numpy as np
from churn_library import (
    import_data,
    perform_eda,
    encoder_helper,
    perform_feature_engineering,
    train_models
)

# Configure logging
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    filename='./logs/test_churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('test_churn_library')


######################### FIXTURES ##################################

@pytest.fixture(scope="module")
def data_path():
    """
    Fixture - Returns the path to the data file for testing.
    """
    return "./data/bank_data.csv"


@pytest.fixture(scope="module")
def invalid_path():
    """
    Fixture - Returns an invalid path for testing error handling.
    """
    return "./data/non_existent_file.csv"


@pytest.fixture(scope="module")
def dataset(data_path):
    """
    Fixture - Returns the imported dataset for use in tests.
    """
    return import_data(data_path)


@pytest.fixture(scope="module",
                params=[
                    # Standard set of categorical features
                    ['Gender', 'Education_Level', 'Marital_Status',
                     'Income_Category', 'Card_Category'],
                    # Subset of categorical features
                    ['Gender', 'Education_Level', 'Marital_Status',
                     'Income_Category'],
                    # Set with non-existent column to test error handling
                    ['Gender', 'Education_Level', 'Marital_Status',
                     'Income_Category', 'Card_Category', 'Non_Existent_Column']
                ])
def encoding_parameters(request):
    """
    Fixture - Parametrized fixture providing different sets of categorical features.
    """
    category_list = request.param

    # Access the dataset from pytest namespace (will be set during test_import)
    data = getattr(pytest, "test_df", None)
    if data is None:
        # If test_df isn't available, import the data directly
        try:
            data = import_data("./data/bank_data.csv")
        except Exception as err:
            logger.error("Could not load data for encoding parameters: %s", err)
            data = pd.DataFrame()  # Empty dataframe as fallback

    return data.copy(), category_list


@pytest.fixture(scope="module")
def engineered_data():
    """
    Fixture - Returns the feature engineered datasets for model training tests.
    """
    # Access the dataset from pytest namespace (will be set during test_import)
    data = getattr(pytest, "test_df", None)
    if data is None:
        # If test_df isn't available, import the data directly
        try:
            data = import_data("./data/bank_data.csv")
        except Exception as err:
            logger.error("Could not load data for feature engineering: %s", err)
            raise

    return perform_feature_engineering(data)


def test_import_success(data_path):
    """
    Test data import for a valid file path.

    Verifies that:
    1. The import function successfully loads the file
    2. The returned dataframe has expected dimensions
    3. The dataframe contains expected columns
    """
    try:
        df = import_data(data_path)
        logger.info("SUCCESS: Data imported successfully from %s", data_path)

        # Store the dataframe in pytest namespace for use in other tests
        pytest.test_df = df

        # Verify the dataframe has rows and columns
        assert df.shape[0] > 0, "Dataframe has no rows"
        assert df.shape[1] > 0, "Dataframe has no columns"

        # Verify key columns exist
        assert 'Churn' in df.columns, "Churn column not found in dataframe"
        assert 'Customer_Age' in df.columns, "Customer_Age column not found in dataframe"

        # Log the dimensions of the dataframe
        logger.info("Dataframe dimensions: %s rows, %s columns", df.shape[0], df.shape[1])

    except AssertionError as err:
        logger.error("FAILED: Data import validation error: %s", err)
        raise
    except Exception as err:
        logger.error("FAILED: Data import error: %s", err)
        raise


def test_import_failure(invalid_path):
    """
    Test data import error handling for an invalid file path.

    Verifies that:
    1. The import function raises FileNotFoundError for an invalid path
    """
    with pytest.raises(FileNotFoundError):
        import_data(invalid_path)
        logger.info("SUCCESS: FileNotFoundError raised for invalid path %s", invalid_path)


def test_eda(dataset):
    """
    Test the exploratory data analysis function.

    Verifies that:
    1. The function runs without errors
    2. The expected output files are created
    """
    try:
        perform_eda(dataset)
        logger.info("SUCCESS: EDA performed successfully")

        # Check that the expected output files exist
        expected_plots = [
            "./images/eda/Customer_Age.png",
            "./images/eda/Total_Trans_Ct.png",
            "./images/eda/correlation_matrix.png",
            "./images/eda/trans_amt_vs_count.png"
        ]

        for plot_path in expected_plots:
            assert os.path.isfile(plot_path), f"Expected plot file not found: {plot_path}"
            logger.info("Plot file found: %s", plot_path)

    except AssertionError as err:
        logger.error("FAILED: EDA validation error: %s", err)
        raise
    except Exception as err:
        logger.error("FAILED: EDA error: %s", err)
        raise


def test_encoder_helper_success(encoding_parameters):
    """
    Test the encoder_helper function with valid parameters.

    For the standard and subset parameter sets, verifies that:
    1. The function runs without errors
    2. All categorical columns are encoded as expected
    3. The encoded columns have the expected naming pattern
    """
    data, category_list = encoding_parameters

    # Skip test cases with non-existent columns
    if any(col not in data.columns for col in category_list):
        pytest.skip("Skipping test with non-existent columns")

    try:
        encoded_df = encoder_helper(data, category_list)
        logger.info("SUCCESS: Encoding completed for categories: %s", category_list)

        # Verify all original categorical columns are dropped
        for category in category_list:
            assert category not in encoded_df.columns, f"Original category {category} not dropped"

        # Verify new encoded columns are created with expected naming pattern
        for category in category_list:
            new_col = f"{category}_Churn"
            assert new_col in encoded_df.columns, f"Expected encoded column {new_col} not found"

            # Verify encoded column has valid values (between 0 and 1 for Churn)
            assert encoded_df[new_col].min() >= 0, f"Encoded column {new_col} has values < 0"
            assert encoded_df[new_col].max() <= 1, f"Encoded column {new_col} has values > 1"

        logger.info("All categorical columns successfully encoded")

    except AssertionError as err:
        logger.error("FAILED: Encoder validation error: %s", err)
        raise
    except Exception as err:
        logger.error("FAILED: Encoder error: %s", err)
        raise


def test_encoder_helper_failure(encoding_parameters):
    """
    Test encoder_helper error handling with invalid parameters.

    For the parameter set with non-existent columns, verifies that:
    1. The function raises KeyError
    """
    data, category_list = encoding_parameters

    # Only run test for cases with non-existent columns
    if all(col in data.columns for col in category_list):
        pytest.skip("Skipping test without non-existent columns")

    with pytest.raises(KeyError):
        encoder_helper(data, category_list)
        logger.info("SUCCESS: KeyError raised for non-existent column in %s", category_list)

def test_perform_feature_engineering(dataset):
    """
    Test the feature engineering function.

    Verifies that:
    1. The function runs without errors
    2. The function returns the expected data structures
    3. The returned data has the expected characteristics
    """
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(dataset)
        logger.info("SUCCESS: Feature engineering performed successfully")

        # Verify output types
        assert isinstance(X_train, pd.DataFrame), "X_train is not a DataFrame"
        assert isinstance(X_test, pd.DataFrame), "X_test is not a DataFrame"
        assert isinstance(y_train, pd.Series), "y_train is not a Series"
        assert isinstance(y_test, pd.Series), "y_test is not a Series"

        # Verify data shapes
        assert X_train.shape[0] > 0, "X_train has no rows"
        assert X_test.shape[0] > 0, "X_test has no rows"
        assert X_train.shape[1] == X_test.shape[1], "X_train and X_test have different numbers of columns"
        assert y_train.shape[0] == X_train.shape[0], "y_train and X_train have different numbers of rows"
        assert y_test.shape[0] == X_test.shape[0], "y_test and X_test have different numbers of rows"

        # Verify expected split ratio (70% train, 30% test)
        total_samples = len(y_train) + len(y_test)
        train_ratio = len(y_train) / total_samples
        test_ratio = len(y_test) / total_samples

        assert 0.65 <= train_ratio <= 0.75, f"Train ratio ({train_ratio:.2f}) outside expected range 0.65-0.75"
        assert 0.25 <= test_ratio <= 0.35, f"Test ratio ({test_ratio:.2f}) outside expected range 0.25-0.35"

        # Verify target distribution
        train_churn_rate = y_train.mean()
        test_churn_rate = y_test.mean()
        overall_churn_rate = dataset['Churn'].mean()

        assert abs(train_churn_rate - overall_churn_rate) < 0.05, \
            f"Train churn rate ({train_churn_rate:.2f}) too different from overall ({overall_churn_rate:.2f})"
        assert abs(test_churn_rate - overall_churn_rate) < 0.05, \
            f"Test churn rate ({test_churn_rate:.2f}) too different from overall ({overall_churn_rate:.2f})"

        logger.info("Train set: %d samples (%.2f%% churn)", len(y_train), train_churn_rate * 100)
        logger.info("Test set: %d samples (%.2f%% churn)", len(y_test), test_churn_rate * 100)

    except AssertionError as err:
        logger.error("FAILED: Feature engineering validation error: %s", err)
        raise
    except Exception as err:
        logger.error("FAILED: Feature engineering error: %s", err)
        raise


def test_train_models(engineered_data):
    """
    Test the train_models function.

    Verifies that:
    1. The function runs without errors
    2. The expected output files (models and plots) are created
    """
    try:
        X_train, X_test, y_train, y_test = engineered_data
        train_models(X_train, X_test, y_train, y_test)
        logger.info("SUCCESS: Models trained successfully")

        # Check that the expected model files exist
        expected_models = [
            "./models/rfc_model.pkl",
            "./models/logistic_model.pkl"
        ]

        for model_path in expected_models:
            assert os.path.isfile(model_path), f"Expected model file not found: {model_path}"
            assert os.path.getsize(model_path) > 0, f"Model file is empty: {model_path}"
            logger.info("Model file found: %s", model_path)

        # Check that the expected result plots exist
        expected_plots = [
            "./images/results/ROC_curves.png",
            "./images/results/Random_Forest_Confusion_Matrix.png",
            "./images/results/Logistic_Regression_Confusion_Matrix.png",
            "./images/results/feature_importance_Random_Forest.png",
            "./images/results/classification_report_Random_Forest.png",
            "./images/results/classification_report_Logistic_Regression.png"
        ]

        for plot_path in expected_plots:
            assert os.path.isfile(plot_path), f"Expected plot file not found: {plot_path}"
            assert os.path.getsize(plot_path) > 0, f"Plot file is empty: {plot_path}"
            logger.info("Result plot found: %s", plot_path)

    except AssertionError as err:
        logger.error("FAILED: Model training validation error: %s", err)
        raise
    except Exception as err:
        logger.error("FAILED: Model training error: %s", err)
        raise


if __name__ == "__main__":
    """
    Main function to run tests individually. This is useful for debugging.
    Note: For regular test runs, use pytest from the command line.
    """
    # Test setup
    data_path = "./data/bank_data.csv"
    invalid_path = "./data/non_existent_file.csv"

    # Run individual tests
    print("Testing import_data with valid path...")
    test_import_success(data_path)

    print("Testing import_data with invalid path...")
    try:
        test_import_failure(invalid_path)
    except Exception as e:
        print(f"Expected error: {e}")

    # Get dataset for remaining tests
    dataset = import_data(data_path)

    print("Testing perform_eda...")
    test_eda(dataset)

    # Define test categories for encoding
    valid_categories = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    invalid_categories = ['Gender', 'Education_Level', 'Non_Existent_Column']

    print("Testing encoder_helper with valid categories...")
    test_encoder_helper_success((dataset.copy(), valid_categories))

    print("Testing encoder_helper with invalid categories...")
    try:
        test_encoder_helper_failure((dataset.copy(), invalid_categories))
    except Exception as e:
        print(f"Expected error: {e}")

    print("Testing perform_feature_engineering...")
    test_perform_feature_engineering(dataset)

    print("Testing train_models...")
    X_train, X_test, y_train, y_test = perform_feature_engineering(dataset)
    test_train_models((X_train, X_test, y_train, y_test))

    print("All tests completed!")


# Sales Prediction Model Using LightGBM
## Overview
This project is a machine learning pipeline for predicting store sales using the LightGBM model. The data used for this project consists of historical sales data, along with additional information such as oil prices, holiday events, and store transactions. The goal is to predict future sales based on this information. The code was built using Python, leveraging popular libraries such as pandas, numpy, scikit-learn, and lightgbm.

## Requirements
To run the code, you need to have the following dependencies installed:
  
{ 
  Python 3.x
  pandas
  numpy
  scikit-learn
  xgboost
  lightgbm
  matplotlib
  seaborn
}

# Dataset

The model requires the following CSV files:

train.csv: Contains the historical sales data.

stores.csv: Contains information about the stores.

oil.csv: Contains the daily oil prices, which may impact sales.

holidays_events.csv: Lists the holidays and events.

transactions.csv: Contains the number of daily transactions for each store.

test.csv: Contains the test data for which the model will predict sales.

Ensure all CSV files are in the same directory as the code or specify the correct path for each file when loading them.

# Features

The features used to train the model include:

store_nbr: Unique identifier for each store.

family: Encoded categorical feature representing the product family.

onpromotion: Number of items on promotion.

dcoilwtico: Daily oil price, filled using forward fill.

transactions: Number of transactions in the store.

type_x: Encoded categorical feature representing the store type.

cluster: Cluster identification of the store.

The target variable is sales, representing the sales amount for each product family at a store on a given date.

# Model Training
## Data Preparation:

The date field is converted to a datetime type.
Data is merged from multiple sources (stores.csv, oil.csv, holidays_events.csv, transactions.csv) into the training and test datasets.
Missing values in onpromotion, dcoilwtico, and transactions are filled with appropriate methods.
Categorical features such as family and type_x are encoded into numerical values using .cat.codes().
## LightGBM Model:

The LightGBM model is initialized with specific parameters for regression.
The model is trained on the features extracted from the training dataset.
The num_boost_round is set to 100, allowing 100 boosting rounds for training.
## Prediction:

After training, predictions are made on the test set, and the results are saved to a submission.csv file.
Negative sales predictions are clipped to a minimum value to avoid errors in the final output.

## Evaluation:

Validation Root Mean Squared Logarithmic Error (RMSLE) is calculated to measure the accuracy of the predictions.

# Usage

Place the required dataset files (train.csv, stores.csv, oil.csv, holidays_events.csv, transactions.csv, and test.csv) in the working directory.
Run the code on a Python environment such as Jupyter Notebook, Google Colab, or your local Python setup.
Once the model is trained, predictions are generated for the test data, and the results are saved to submission.csv.

# Results
The final output is a CSV file (submission.csv) containing the predicted sales for the test data, with the following columns:

id: The unique identifier of the test data row.

sales: The predicted sales value for the corresponding row.


# Model Performance

The model's performance is evaluated on the validation data using the Root Mean Squared Logarithmic Error (RMSLE).

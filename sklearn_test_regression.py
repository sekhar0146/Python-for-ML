# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:51:15 2020

@author: z011348
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in the car sales data
car_sales = pd.read_csv("C:/Users/z011348/Desktop/ML/input/car-sales-extended-missing-data.csv")
print(car_sales.head())
"""
How many rows are there total?
What datatypes are in each column?
How many missing values are there in each column?
"""
print("shape of DF : ", car_sales.shape)  # rows, columms 
print("")
print("Car_sales data types : ", car_sales.dtypes) # data types
print("")
print("null/missing values in the DF:")

# Import the data and drop missing labels from target coloumn 
car_sales.dropna(subset=["Price"], inplace=True)

print(car_sales.isna().sum())

# Import Pipeline from sklearn's pipeline module
from sklearn.pipeline import Pipeline

# Import ColumnTransformer from sklearn's compose module
from sklearn.compose import ColumnTransformer

# Import SimpleImputer from sklearn's impute module
from sklearn.impute import SimpleImputer

# Import OneHotEncoder from sklearn's preprocessing module
from sklearn.preprocessing import OneHotEncoder

# Import train_test_split from sklearn's model_selection module
from sklearn.model_selection import train_test_split

# Define different categorical features 
categorical_features = ["Make", "Colour"]
# Create categorical transformer Pipeline
categorical_transformer = Pipeline(steps=[
    # Set SimpleImputer strategy to "constant" and fill value to "missing"
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    # Set OneHotEncoder to ignore the unknowns
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

# Door features
door_feature = ["Doors"]
# Create door transformer Pipeline
door_transformer = Pipeline(steps=[
    # Set SimpleImputer strategy to "constant" and fill value to 4
    ("imputer", SimpleImputer(strategy="constant", fill_value=4))
    ])

# Define numeric features (only the Odometer (KM) column)
numeric_features = ["Odometer (KM)"]
# Create numeric transformer Pipeline
numeric_transformer = Pipeline(steps=[
    # Set SimpleImputer strategy to fill missing values with the "Median"
    ("imputer", SimpleImputer(strategy="median"))
    ])

# Setup preprocessing steps (fill missing values, then convert to numbers)
preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_transformer, categorical_features),
    ("door", door_transformer, door_feature),
    ("num", numeric_transformer, numeric_features)
    ])

# Import Ridge from sklearn's linear_model module
from sklearn.linear_model import Ridge
# Import SVR from sklearn's svm module
from sklearn.svm import SVR
# Import RandomForestRegressor from sklearn's ensemble module
from sklearn.ensemble import RandomForestRegressor

# Create dictionary of model instances, there should be 4 total key, value pairs
# in the form {"model_name": model_instance}.
# Don't forget there's two versions of SVR, one with a "linear" kernel and the
# other with kernel set to "rbf".
reg_models = {"Ridge": Ridge(),
              "SVR_linear": SVR(kernel="linear"),
              "SVR_rbf": SVR(kernel="rbf"),
              "RandomForestRegressor": RandomForestRegressor()}

reg_results={}

# Split the data into X and y
X = car_sales.drop("Price", axis=1)
y = car_sales["Price"]

# Use train_test_split to split the car_sales_X and car_sales_y data into 
# training and test sets.
# Give the test set 20% of the data using the test_size parameter.
# For reproducibility set the random_state parameter to 42.
# np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print("=== Check the shapes after train === ")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("")
 
for model_name, model in reg_models.items():
    # Create a model pipeline with a preprocessor step and model step
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)])
    # Fit the model Pipeline to the car sales training data
    print(f"Fitting {model_name}... ")
    model_pipeline.fit(X_train, y_train)
    # Score the model Pipeline on the test data appending the model_name to the 
    # results dictionary
    print(f"Scoring {model_name}... ")
    reg_results[model_name] = model_pipeline.score(X_test, y_test)
  
# Check the results of each regression model by printing the regression_results
# dictionary
print("")
print(reg_results)
print("")
# Import mean_absolute_error from sklearn's metrics module
from sklearn.metrics import mean_absolute_error

# Import mean_squared_error from sklearn's metrics module
from sklearn.metrics import mean_squared_error

# Import r2_score from sklearn's metrics module
from sklearn.metrics import r2_score

# Create RidgeRegression Pipeline with preprocessor as the "preprocessor" and
# Ridge() as the "model".
ridge_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge())
    ])

# Fit the RidgeRegression Pipeline to the car sales training data
ridge_pipeline.fit(X_train, y_train)

# Make predictions on the car sales test data using the RidgeRegression Pipeline
y_pred = ridge_pipeline.predict(X_test)

# View the first 50 predictions
print("View the first 50 predictions:")
print(y_pred[:50])
print("")

# Find the MSE by comparing the car sales test labels to the car sales predictions
mse = mean_squared_error(y_test, y_pred)
print("mse: ", mse)

# Find the MAE by comparing the car sales test labels to the car sales predictions
mae = mean_absolute_error(y_test, y_pred)
print("mae: ", mae)

# Find the R^2 score by comparing the car sales test labels to the car sales predictions
r2 = r2_score(y_test, y_pred)
print("r2_score : ", r2)
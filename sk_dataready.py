
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

heart_disease = pd.read_csv('C:/Users/Desktop/ML/input/heart-disease.csv')
car_sales = pd.read_csv('C:/Users/Desktop/ML/input/car-sales-extended.csv')

X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# split data into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
# test_size = 0.2 ==> considering 20% data for test
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# -------------------------------------------------- 
# CAR_SALES 
# Make sure all numerical values 
# Convert them if not 
# -------------------------------------------------- 
print(len(car_sales))
print(car_sales.dtypes)
X = car_sales.drop("Price", axis=1)
y = car_sales["Price"]
print(X)
print(y)
# split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# build machine learning model
# gives an error 
# ValueError: could not convert string to float: 'Honda'
"""
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)
"""

# now we will convert strings into numerics
# One way to convert - OnehotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

features = ["Make", "Colour", "Doors"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                  one_hot,
                                  features)],    
                                  remainder="passthrough")
transformed_X = transformer.fit_transform(X)
#print(transformed_X)
car_sales_num = pd.DataFrame(transformed_X)
print(car_sales_num)

# One way to convert - dummy
dummies = pd.get_dummies(car_sales[["Make", "Colour", "Doors"]])
print(dummies)

# Now let fit the model
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(transformed_X,
                                                    y,
                                                    test_size=0.2)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))


# -------------------------------------------------- 
# Dealing missing values
# -------------------------------------------------- 
car_sales_missing = pd.read_csv('C:/Users/Desktop/ML/input/car-sales-extended-missing-data.csv')
print(car_sales_missing.isna().sum())
print(" ")
# Fill missing data with Pandas
car_sales_missing["Make"].fillna("missing", inplace=True)
car_sales_missing["Colour"].fillna("missing", inplace=True)
car_sales_missing["Odometer (KM)"].fillna(car_sales_missing["Odometer (KM)"].mean(), inplace=True)
car_sales_missing["Doors"].fillna(4, inplace=True)
print(car_sales_missing.isna().sum())
print(" ")
# Remove rows with missing Price value
car_sales_missing.dropna(inplace=True)
print(car_sales_missing.isna().sum())
print(len(car_sales_missing))

# Split data into features and labels
X = car_sales_missing.drop("Price", axis=1)
y = car_sales_missing["Price"]

features = ["Make", "Colour", "Doors"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                  one_hot,
                                  features)],    
                                  remainder="passthrough")
transformed_X = transformer.fit_transform(car_sales_missing)
#print(transformed_X)
car_sales_missing_num = pd.DataFrame(transformed_X)
print(car_sales_missing_num)


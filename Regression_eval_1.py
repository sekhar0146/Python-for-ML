"""
Regression model evaluations:
	- R^2 (pronounced r-squared) or coefficient of determination
	- Mean absolute error (MAE)
	- Mean squared error (MSE)
"""

# --------------------------------------------------------------
# R^2 (pronounced r-squared) or coefficient of determination
# --------------------------------------------------------------

import numpy as np
import pandas as pd

# import bostun housing data set

from sklearn.datasets import load_boston
boston = load_boston()
#print(boston)
boston_df = pd.DataFrame(
    boston["data"],
    columns=boston["feature_names"])
boston_df["target"] = pd.Series(boston["target"])

np.random.seed(42)

# Craete the data
X = boston_df.drop("target", axis=1)
y = boston_df["target"]

# split data into training and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)

# Instantiate RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
rr = RandomForestRegressor()
rr.fit(X_train, y_train)

#----------------------------
# Score function  
#----------------------------
rr_single_score = rr.score(X_test, y_test)
print(rr_single_score)

# ----------------------------------
# R^2 
# ----------------------------------
from sklearn.metrics import r2_score
y_test_mean = np.full(len(y_test), y_test.mean())
# print(y_test_mean)
print(y_test.mean())
print(" ")
print("R2 is : ")
print(r2_score(y_test, y_test_mean))
print(r2_score(y_test, y_test))


# ----------------------------------
# Mean absolute error (MAE)
# ----------------------------------
from sklearn.metrics import mean_absolute_error
y_pred = rr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("MAE is : ")
print(mae)

# check the data between actual and predict
df = pd.DataFrame(data={"Actual values": y_test,
                        "Predicted values": y_pred})
df["difference"] = df["Predicted values"] - df["Actual values"]
print(df)

# ----------------------------------
# Mean squared error (MSE)
# ----------------------------------
from sklearn.metrics import mean_squared_error
y_pred = rr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE value is : ", mse)

# manual calculation of mean square 
sq = np.square(df["difference"])
mean_square = sq.mean()
print("Mean square by Manual is : ", mean_square)

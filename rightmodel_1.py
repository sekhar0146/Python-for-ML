# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:06:13 2020

@author: z011348
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# import bostun housing data set
from sklearn.datasets import load_boston
boston = load_boston()
#print(boston)
boston_df = pd.DataFrame(
    boston["data"],
    columns=boston["feature_names"])
boston_df["target"] = pd.Series(boston["target"])
#print(boston_df)
print(len(boston_df))

# =============================================
# lets try the Redge regression model
# =============================================
from sklearn.linear_model import Ridge

# setup random see
np.random.seed(42)

# Create the data
X = boston_df.drop("target", axis=1)
y = boston_df["target"]

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
# Instantiate Ridge model
model = Ridge()
model.fit(X_train, y_train)

# Check the score
print(model.score(X_test, y_test))      # 0.6662221670168522

# =============================================
# lets try the RandomForest model
# =============================================
from sklearn.ensemble import RandomForestRegressor

# setup random see
np.random.seed(42)

# Create the data
# X = boston_df.drop("target", axis=1)
# y = boston_df["target"]

# Split train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X,
#                                                    y,
#                                                    test_size=0.2)
# Instantiate Ridge model
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Check the score
print(rf.score(X_test, y_test))     #0.873969014117403

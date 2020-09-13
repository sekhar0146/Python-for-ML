
import numpy as np
import pandas as pd
heart_disease = pd.read_csv('C:/Users/Desktop/ML/input/heart-disease.csv')

# setup random see
np.random.seed(42)

print(" RC ")
# make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# split data into training and test data
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
# Instantiate RandomForestClassifier 
rc = RandomForestClassifier()

# Fit the model
rc.fit(X_train, y_train)

# Three ways to evaluate scikit learn models
# 1. Estimator score method
# 2. The score parameter
# 3. Problem specific metric functions 

#----------------------------
# Score function  
#----------------------------
print(rc.score(X_train, y_train))

rc_single_score = rc.score(X_test, y_test)
print(rc.score(X_test, y_test))

#-------------------------------------------------
# Scoring paramter - 5 fold cross val - Accuracy
#-------------------------------------------------
from sklearn.model_selection import cross_val_score

rc_cross_val_score = np.mean(cross_val_score(rc, X, y, cv=5))
print(cross_val_score(rc, X, y, cv=5, scoring=None))    # scoring=None is the default 
# print(cross_val_score(rc, X, y, cv=10))

# Compare 2 Scores 
print(rc_single_score, rc_cross_val_score)
print(f"Heart Disease classifier cross-validation accuracy : {np.mean(rc_cross_val_score) * 100:.2f}%")

print("")
print(" RR ")
#########################
# RandonForestRegressor
#########################
# import bostun housing data set
np.random.seed(42)

from sklearn.datasets import load_boston
boston = load_boston()
#print(boston)
boston_df = pd.DataFrame(
    boston["data"],
    columns=boston["feature_names"])
boston_df["target"] = pd.Series(boston["target"])

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
print("rc_single_score = ", rc_single_score,
      ";", 
      "rc_cross_val_score = ", rc_cross_val_score)


#----------------------------
# Scoring paramter 
#----------------------------
from sklearn.model_selection import cross_val_score

rr_cross_val_score = np.mean(cross_val_score(rr, X, y, cv=5))
print(cross_val_score(rr, X, y, cv=5, scoring=None))    # scoring=None is the default 
# print(cross_val_score(rc, X, y, cv=10))

# Compare 2 Scores 
print("rr_single_score = ", rr_single_score,
      ";", 
      "rr_cross_val_score = ", rr_cross_val_score)

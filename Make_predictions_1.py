# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 08:08:32 2020

@author: z011348
"""
import numpy as np
import pandas as pd
heart_disease = pd.read_csv('C:/Users/z011348/Desktop/ML/input/heart-disease.csv')

# setup random see
np.random.seed(42)

# make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# split data into training and test data
from sklearn.model_selection import train_test_split

# =============================================
# lets try the RandomForest classifier
# =============================================
from sklearn.ensemble import RandomForestClassifier

# split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
# Instantiate RandomForestClassifier()
rc = RandomForestClassifier()
rc.fit(X_train, y_train)

# Check/evaluate the model
print(rc.score(X_test, y_test))     # 0.8524590163934426

# Use trained model to make the predictions - predict()
y_pred = rc.predict(X_test)
print(y_pred[:5])

# Compare prdictions to truth lables to evaluate the model - one way
print(np.mean(y_pred == y_test))

# Compare prdictions to truth lables to evaluate the model - another way
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Use trained model to make the predictions - predict_proba()
# predict_proba() returns probabilities of a classification label
y_pred_proba = rc.predict_proba(X_test[:5])
print(y_pred_proba)


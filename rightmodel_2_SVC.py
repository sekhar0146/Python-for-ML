# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 17:43:43 2020

@author: z011348
"""

import numpy as np
import pandas as pd

heart_disease = pd.read_csv('C:/Users/z011348/Desktop/ML/input/heart-disease.csv')
#print(len(heart_disease))

# =============================================
# lets try the liner support vector model
# =============================================
#Import liner support vector classification 
from sklearn.svm import LinearSVC

# Setup random seed
np.random.seed(42)

# make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# split data into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
# Instantiate linear SVC
clf = LinearSVC()
clf.fit(X_train, y_train)

# Check/evaluate the model
print(clf.score(X_test, y_test))        # 0.4918032786885246

# =============================================
# lets try the RandomForest classifier
# =============================================
# setup random see
np.random.seed(42)

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


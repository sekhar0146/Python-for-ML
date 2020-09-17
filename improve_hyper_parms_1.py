# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:11:50 2020

@author: z011348
"""

import numpy as np
import pandas as pd

def evaluate_preds(y_test, y_pred):
    """
    performance evaluation comparison on y_true lables and y_pred labels
    on classification
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mrtric_dict ={"accuracy": round(accuracy,2),
                  "precision": round(precision,2),
                  "recall": round(recall,2),
                  "f1": round(f1,2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:2f}")
    print(f"F1: {f1:2f}")

# 
heart_disease = pd.read_csv('C:/Users/z011348/Desktop/ML/input/heart-disease.csv')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
np.random.seed(34)

# Shuffle the data
heart_disease_shuffled = heart_disease.sample(frac=1)

# Split into X and y
X = heart_disease_shuffled.drop("target", axis=1)
y = heart_disease_shuffled["target"]

# Split the data into Train, validation and test data sets
# 70% data
train_split = round(0.7 * len(heart_disease_shuffled))  
# 15% data
valid_split = round(train_split + 0.15 * len(heart_disease_shuffled))  # 70% data
X_train, y_train = X[:train_split], y[:train_split]
X_valid, y_valid = X[train_split:valid_split], y[train_split:valid_split]
X_test, y_test = X[valid_split:], y[valid_split:]

print(len(X_train), len(X_valid), len(X_test))

# Instantiate RandomForestClassifier
clf = RandomForestClassifier()

# Fit the model
clf.fit(X_train, y_train)

# Make baseline predictions
y_pred = clf.predict(X_valid)

# Evaluate classifier on validation set
print(" == baseline metrics ==")
baseline_metrics = evaluate_preds(y_valid, y_pred)

#----
# Second classifier with n_estimators
#----
np.random.seed(34)
# Instantiate RandomForestClassifier
clf2 = RandomForestClassifier(n_estimators=20)

# Fit the model
clf2.fit(X_train, y_train)

# Make baseline predictions
y_pred = clf2.predict(X_valid)

# Evaluate classifier on validation set
print(" == 2nd classifier metrics ==")
clf2_metrics = evaluate_preds(y_valid, y_pred)

#----
# 3rd classifier with max_depth
#----
np.random.seed(34)
# Instantiate RandomForestClassifier
clf3 = RandomForestClassifier(n_estimators=20,
                              max_depth=20)

# Fit the model
clf3.fit(X_train, y_train)

# Make baseline predictions
y_pred = clf3.predict(X_valid)

# Evaluate classifier on validation set
print(" == 3rd classifier metrics ==")
clf3_metrics = evaluate_preds(y_valid, y_pred)


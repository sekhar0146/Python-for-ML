# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:22:59 2020

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
    
    
heart_disease = pd.read_csv('C:/Users/z011348/Desktop/ML/input/heart-disease.csv')

#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from sklearn.ensemble import RandomForestClassifier

# Shuffle the data
heart_disease_shuffled = heart_disease.sample(frac=1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

grid_2 ={"n_estimators": [100, 200, 500],
       "max_depth": [None],
       "max_features": ["auto", "sqrt"],
       "min_samples_split": [6],
       "min_samples_leaf":[1,2]}

np.random.seed(34)

# Split into X and y
X = heart_disease_shuffled.drop("target", axis=1)
y = heart_disease_shuffled["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)

# Instantiate the RandomForestClassifier
clf = RandomForestClassifier(n_jobs=-1)

# Setup GridSearchCV
gs_clf = GridSearchCV(estimator=clf,
                            param_grid=grid_2,
                            cv=5,
                            verbose=2) 

# Fit the GridSearchCV version of clf
gs_clf.fit(X_train, y_train)
 
# Get the best parameters
print(" ================ best_params_ ===================")
print(gs_clf.best_params_)

# Make predictions with the best hyperparameters
gs_y_pred = gs_clf.predict(X_test)

# Evaluate the predicions
print(" ================ Evaluation metrics ===================")
gs_metrics = evaluate_preds(y_test, gs_y_pred)

# =======================================
# Saving the model
# =======================================
import pickle

# save the existing model to file
pickle.dump(gs_clf, open("C:/Users/z011348/Desktop/ML/output/GridSearch_RandomForest_model.pkl", "wb" ))

# -------
# load the saved model with joblib

from joblib import dump, load
dump(gs_clf, filename="C:/Users/z011348/Desktop/ML/output/GridSearch_RandomForest_model.joblib")

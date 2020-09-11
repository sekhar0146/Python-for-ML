# SKLEARN workflow

import pandas as pd
import numpy as np

# ==> Warnings to ignore
#import warnings
#warnings.filterwarnings("ignore")

heart_disease = pd.read_csv('C:/Users/Desktop/ML/input/heart-disease.csv')

# create x (Features matrix)
x = heart_disease.drop("target", axis=1)

# create y (Labels)
y = heart_disease["target"]

# choose the right model and hyperparameters
from sklearn.ensemble import RandomForestClassifier
# RandomForestClassifier - classification ML model
clf = RandomForestClassifier(n_estimators=40)
#==> n_estimators=40 have given after improving the model

# we will keep the default hyperparameters
clf.get_params()
print(clf.get_params())

# Fit the model to the training data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2) 

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
# print(y_predicts)
# print(y_test)

# Evaluate the model on training data and test data
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
clf.score(X_train, y_train)

# metrics
from sklearn.metrics import classification_report,  confusion_matrix, accuracy_score

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# improve the model
# try different amount of n_estimators 
np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with {i} estimators .. ")
    clf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    print(f"Model accuracy on test set: {clf.score(X_test, y_test) * 100:.2f}%")
    print(" ")

# Save the model
import pickle
pickle.dump(clf, open("C:/Users/Desktop/ML/output/random_forest_model_1.pkl", "wb"))

loaded_model = pickle.load(open("C:/Users/Desktop/ML/output/random_forest_model_1.pkl", "rb"))
print(loaded_model.score(X_test, y_test))

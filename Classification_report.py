import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Classification report way to compare the labels a model predicts
# and the actual label it was supposed to predict 
# -----------------------------------------------------------------------

# 
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

# predict the labels
y_pred = rc.predict(X_test)
print(y_pred)

# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

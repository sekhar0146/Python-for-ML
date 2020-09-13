
import numpy as np
import pandas as pd

# 
heart_disease = pd.read_csv('C:/Users/Desktop/ML/input/heart-disease.csv')

#=====================================================================
# Different evaluation metrics using scikit learn
# Classification evaluation metrics
#=====================================================================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(34)

# make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# make some predictions
y_pred = clf.predict(X_test)

print("=========== Classification evaluation metrics ===========")
# Evaluate classifier 
print("Classifier metrics on the test set")
print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.02f}")
print(f"Precision : {precision_score(y_test, y_pred)}")
print(f"Recall : {recall_score(y_test, y_pred)}")
print(f"F1 : {f1_score(y_test, y_pred)}")

# ==========================================================
# Regression evaluation metrics
# ==========================================================
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
print("=========== Regression evaluation metrics ===========")

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

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# make some predictions
y_pred = model.predict(X_test)

# Evaluate Rgressor
print("Regressor metrics on the test set")
print(f"R^2 : {r2_score(y_test, y_pred)}")
print(f"MAE : {mean_absolute_error(y_test, y_pred)}")
print(f"MSE : {mean_squared_error(y_test, y_pred)}")





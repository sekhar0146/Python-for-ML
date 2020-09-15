import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Area under the receiver operating characterstic curse (AUC/ROC)
# Area under curve (AUC)
# ROC curve
# 
# ROC curves are comparison of model's true positive rate (tpr) vs
# a model false positive rate (fpr) 
# 
# True positive - model predicts 1 when truth is 1
# False positive - model predicts 1 when truth is 0
# True negative - model predicts 0 when truth is 0
# False negative - model predicts 0 when truth is 1
# ------------------------------------------------------------------

# Make predictions with probabilities
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


# Use trained model to make the predictions - predict_proba()
# predict_proba() returns probabilities of a classification label
y_pred_proba = rc.predict_proba(X_test)
#print(y_pred_proba)
# print(len(y_pred_proba))

# get the data related to '1'
y_prob_positive = y_pred_proba[:, 1]
print(y_prob_positive[:10])

from sklearn.metrics import roc_curve
# Calculate fpr, tpr and thresholds 
fpr, tpr, thesholds = roc_curve(y_test, y_prob_positive)
# print(fpr)


# Visual representation 
import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr):
    
    # plot roc curve
    plt.plot(fpr, tpr, color='orange', label='ROC')
    # plot line with no predictive power (baseline)
    plt.plot([0,1], [0,1], color='darkblue', linestyle='--', label='Guessing')
    
    # customize plots
    plt.xlabel("False predictive rate(fpr)")
    plt.ylabel("True predictive rate(tpr)")
    plt.title("Receiver operating characterstic curse (ROC ")
    plt.legend()
    plt.show()
    
plot_roc_curve(fpr, tpr)

# check the accurate score
from sklearn.metrics import roc_auc_score
print("Prob positive score is : ")
print(roc_auc_score(y_test, y_prob_positive))

print('')
# plot perfect ROC curve and AUC score
fpr, tpr, thesholds = roc_curve(y_test, y_test)
plot_roc_curve(fpr, tpr)
print(roc_auc_score(y_test, y_test))



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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

grid ={"n_estimators": [10, 100, 200, 500, 1000, 1200],
       "max_depth": [None, 5, 10, 20, 30],
       "max_features": ["auto", "sqrt"],
       "min_samples_split": [2,4,6],
       "min_samples_leaf":[1,2,4]}

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

# Setup RandomizedSearchCV
rs_clf = RandomizedSearchCV(estimator=clf,
                            param_distributions=grid,
                            n_iter=10,  # Number of models to try
                            cv=5,
                            verbose=2) 

# Fit the RandomizedSearchCV version of clf
rs_clf.fit(X_train, y_train)
 
# Get the best parameters
print(" ================ best_params_ ===================")
print(rs_clf.best_params_)

# Make predictions with the best hyperparameters
rs_y_pred = rs_clf.predict(X_test)

# Evaluate the predicions
print(" ================ Evaluation metrics ===================")
rs_metrics = evaluate_preds(y_test, rs_y_pred) 

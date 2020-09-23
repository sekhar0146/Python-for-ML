
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

heart_disease = pd.read_csv("C:/Users/z011348/Desktop/ML/input/heart-disease.csv")
# print(heart_disease.head())

# Set the Random seed
np.random.seed(42)

# Create X and y data sets
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data sets into Training and Test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
# print(X_train.shape)    
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# Import the RandomForestClassifier from sklearn's ensemble module
from sklearn.ensemble import RandomForestClassifier

# Instantiate an instance of RandomForestClassifier as clf
clf = RandomForestClassifier()

# Fit the RandomForestClassifier to the training data
clf.fit(X_train, y_train)

# Use the fitted model to make predictions on the test data and
# save the predictions to a variable called y_preds
y_preds = clf.predict(X_test)
print(y_preds)

# Evaluate the fitted model on the training set using the score() function
print(clf.score(X_train, y_train))

# Evaluate the fitted model on the test set using the score() function
print(clf.score(X_test, y_test))

# =======================================================
# Experimenting with different classification models
# =======================================================
print(" == Experimenting with different classification models ==")
"""
LinearSVC
KNeighborsClassifier (also known as K-Nearest Neighbors or KNN)
SVC (also known as support vector classifier, a form of support vector machine)
LogisticRegression (despite the name, this is actually a classifier)
RandomForestClassifier (an ensemble method and what we used above)
"""
"""
Import a machine learning model
Get it ready
Fit it to the data and make predictions
Evaluate the fitted model
"""
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

"""
To see which model performs best, we'll do the following:

Instantiate each model in a dictionary
Create an empty results dictionary
Fit each model on the training data
Score each model on the test data
Check the results
"""
# EXAMPLE: Instantiating a RandomForestClassifier() in a dictionary
example_dict = {"RandomForestClassifier": RandomForestClassifier()}
models ={
    "LinearSVC": LinearSVC(),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(),
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier()
    }

example_result = {}

# EXAMPLE: Looping through example_dict fitting and scoring the model
for model_name, model in example_dict.items():
    model.fit(X_train, y_train)
    example_result[model_name] = model.score(X_test, y_test)
 
# View the results
print(example_result)

result = {}
# Loop through the models dictionary items, fitting the model on the training data
# and appending the model name and model score on the test data to the results dictionary

for model_name, model in models.items():
    model.fit(X_train, y_train)
    result[model_name] = model.score(X_test, y_test)

print(result)    

# Create a pandas dataframe with the data as the values of the results dictionary,
# the index as the keys of the results dictionary and a single column called accuracy.
# Be sure to save the dataframe to a variable.
result_df = pd.DataFrame(result.values(),
                         result.keys(),
                         columns=["accuracy"])
print(result_df)

# Create a bar plot of the results dataframe using plot.bar()
result_df.plot(kind="bar")
plt.show()

# =========================================================
# Hyperparameter Tuning
# =========================================================
log_reg_grid = {"C":np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# import RandomizedSearchCV 
from sklearn.model_selection import RandomizedSearchCV

np.random.seed(42)

# Setup an instance of RandomizedSearchCV with a LogisticRegression() estimator,
# our log_reg_grid as the param_distributions, a cv of 5 and n_iter of 5.
rs_log_reg = RandomizedSearchCV(estimator=LogisticRegression(), 
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=5,
                                verbose=2)

# Fit the instance of RandomizedSearchCV
rs_log_reg.fit(X_train, y_train)

# Find the best parameters of the RandomizedSearchCV instance 
# using the best_params_ attribute
print("== Identify the best hyperparameters for LogisticRegression() ==")
print(rs_log_reg.best_params_)

# Score the instance of RandomizedSearchCV using the test data
print("== Score after tuning  the model == ")
print(rs_log_reg.score(X_test, y_test))

# ==============================================
# Classifier model evaluation 
# ==============================================
clf = LogisticRegression(solver='liblinear', C=0.23357214690901212)
clf.fit(X_train, y_train)

# Import confusion_matrix and classification_report from sklearn's metrics module
from sklearn.metrics import confusion_matrix

# Import precision_score, recall_score and f1_score from sklearn's metrics module
from sklearn.metrics import precision_score, recall_score, f1_score

# Import plot_roc_curve from sklearn's metrics module
from sklearn.metrics import plot_roc_curve

# Import classification_report
from sklearn.metrics import classification_report

# Make predictions on test data and save them
y_pred = clf.predict(X_test)

# Create a confusion matrix using the confusion_matrix function
print(confusion_matrix(y_test, y_pred))

# visualize confusion matrix with pd.crosstab()
df = pd.crosstab(y_test,
                 y_pred,
                 rownames=["Actual labels"],
                 colnames=["Predicted labels"])
print(df)

# Make our prediction more visualize with seaborn heatmap()
import seaborn as sns

# Set the font scale 
sns.set(font_scale=1.5)

# ........................
# Create confusion matrix
# ........................
conf_mat = confusion_matrix(y_test, y_pred)

def plot_conf_mat(conf_mat):
    """
    Plots a confusion matrix using Seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(conf_mat,
                     annot=True,    # annotate the boxes with conf_mat info
                     cbar=False)
    
    plt.xlabel("True label")
    plt.ylabel("Predicted label")

# plot using seaborn
plot_conf_mat(conf_mat)

# .............................
# classification report
# .............................
# Create a classification report using the classification_report function
print(classification_report(y_test, y_pred))

# Find the precision score of the model using precision_score()
print("precision_score : ", precision_score(y_test, y_pred))

# Find the recall score
print("recall_score : ", recall_score(y_test, y_pred))

# Find the F1 score
print("f1_score : ", f1_score(y_test, y_pred))

# Plot a ROC curve using our current machine learning model using plot_roc_curve
print(plot_roc_curve(clf, X_test, y_test))

# .............................
# cross validation
# .............................

# Import cross_val_score from sklearn's model_selection module
from sklearn.model_selection import cross_val_score
print("clf_cross_val_score : ")
clf_cross_val_score = cross_val_score(clf,
                                      X,
                                      y,
                                      scoring="accuracy",
                                      cv=5)
print(clf_cross_val_score)

print("clf_cross_val_score_mean : ")
clf_cross_val_score_mean = np.mean(cross_val_score(clf,
                                                   X,
                                                   y,
                                                   scoring="accuracy",
                                                   cv=5))
print(clf_cross_val_score_mean)

# Find the cross-validated precision
print("clf_cross_val_score_precision : ")
clf_cross_val_score_precision = np.mean(cross_val_score(clf,
                                                   X,
                                                   y,
                                                   scoring="precision",
                                                   cv=5))
print(clf_cross_val_score_precision)

# Find the cross-validated recall
print("clf_cross_val_score_recall : ")
clf_cross_val_score_recall = np.mean(cross_val_score(clf,
                                                   X,
                                                   y,
                                                   scoring="recall",
                                                   cv=5))
print(clf_cross_val_score_recall)

# Find the cross-validated F1 score
print("clf_cross_val_score_f1_score : ")
clf_cross_val_score_f1_score = np.mean(cross_val_score(clf,
                                                   X,
                                                   y,
                                                   scoring="f1",
                                                   cv=5))
print(clf_cross_val_score_f1_score)

# Import the dump and load functions from the joblib library
###
from joblib import dump, load

# Use the dump function to export the trained model to file
###
dump(clf, "C:/Users/z011348/Desktop/ML/output/trained-classifier-test.joblib")

# Use the load function to import the trained model you just exported
# Save it to a different variable name to the origial trained model
loaded_clf = load("C:/Users/z011348/Desktop/ML/output//trained-classifier-test.joblib")

# Evaluate the loaded trained model on the test data
print("Evaluate the loaded trained model on the test data:")
print(loaded_clf.score(X_test, y_test))

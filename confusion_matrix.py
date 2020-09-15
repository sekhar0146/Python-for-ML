import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Confusion matrix is a quick way to compare the labels a model predicts
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

# confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = rc.predict(X_test)
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

# Create confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

# plot using seaborn
#sns.heatmap(conf_mat)

# import matplotlib
import matplotlib.pyplot as plt
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
    

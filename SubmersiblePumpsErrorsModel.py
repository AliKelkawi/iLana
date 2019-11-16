# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:15:06 2019

@author: AliKelkawi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 01:32:13 2019

@author: AliKelkawi
"""

# The dataset 
from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.utils import resample
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from imblearn import over_sampling as os
from imblearn import pipeline as pl
from imblearn.metrics import classification_report_imbalanced



# Importing the dataset
dataset = pd.read_csv('errorsTable.csv')
X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values
feature_names = list(dataset.columns)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
onehotencoder = OneHotEncoder(categorical_features = [4])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:] #Always take one dummy variable away

# https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/
# separate minority and majority classes
# apply near miss 
from imblearn.under_sampling import NearMiss 
nr = NearMiss() 

X, y = nr.fit_sample(X, y.ravel())

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
RFclassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RFclassifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = RFclassifier.predict(X_test)

# Making the Confusion Matrix for RF
from sklearn.metrics import confusion_matrix
cmRF = confusion_matrix(y_test, y_pred)

from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier

clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
clfbag = BaggingClassifier(clf, n_estimators=5)
clfbag.fit(X_train, y_train)
ypreds = clfbag.predict_proba(X_test)
print("loss WITHOUT calibration : ", log_loss(y_test, ypreds, eps=1e-15, normalize=True))

clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
calibrated_clf.fit(X_train, y_train)
ypreds = calibrated_clf.predict_proba(X_test)
print("loss WITH calibration : ", log_loss(y_test, ypreds, eps=1e-15, normalize=True))

y_pred = calibrated_clf.predict(X_test)

#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(RFclassifier, X=X_train, y=y_train, cv=10, scoring='accuracy')
#print('CV accuracy scores: %s' % scores)
#print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#######

def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))
def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0)) 
def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))
def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))

def find_conf_matrix_values(y_true,y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true,y_pred)
    FN = find_FN(y_true,y_pred)
    FP = find_FP(y_true,y_pred)
    TN = find_TN(y_true,y_pred)
    return TP,FN,FP,TN
def my_confusion_matrix(y_true, y_pred):
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return np.array([[TN,FP],[FN,TP]])

myCM = my_confusion_matrix(y_test, y_pred)


from sklearn.metrics import roc_curve
fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, y_pred)


from sklearn.metrics import roc_auc_score
auc_RF = roc_auc_score(y_test, y_pred)

plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Metrics

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

def specificity(confusion_matrix):
    TN = confusion_matrix[0, 0]
    sum_of_TNFP = sum(confusion_matrix[0, :])
    return TN/sum_of_TNFP

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
cmToTest = cmRF
print("Accuracy: " + str(accuracy(cmToTest)))
print("Precision: " + str(precision_score(y_test, y_pred)))
print("Recall: " + str(recall_score(y_test, y_pred)))
print("Specificity: " + str(specificity(cmToTest)))


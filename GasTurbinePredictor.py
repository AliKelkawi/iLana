# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:05:16 2019

@author: AliKelkawi
"""

# Random Forest Regression

# Importing the libraries
import numpy as np
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('Dataset (Gas Turbine Compressor Predictive Maintenance).csv')
X = dataset.iloc[:, 0:16].values
y = dataset.iloc[:, 16].values
feature_names = list(dataset.columns)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)

### COMPRESSOR HEALTH
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Saving model to disk
pickle.dump(regressor, open('CompressorHealthModel.pkl','wb'))


### TURBINE HEALTH
# Fitting Random Forest Regression to the dataset
X = dataset.iloc[:, 0:16].values
y = dataset.iloc[:, 17].values
feature_names = list(dataset.columns)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set Results 
y_pred = regressor.predict(X_test)

# Saving model to disk
pickle.dump(regressor, open('TurbineHealthModel.pkl','wb'))

import sklearn.metrics as metrics
def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', r2)
    print('MAE: ', mean_absolute_error)
    print('MSE: ', mse)
    print('RMSE: ', np.sqrt(mse))
    
regression_results(y_test, y_pred)
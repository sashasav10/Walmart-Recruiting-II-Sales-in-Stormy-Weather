from statistics import LinearRegression

import pandas as pd
import numpy as np
import preprocessor
from sklearn import datasets, linear_model
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


x = pd.read_csv('https://raw.githubusercontent.com/sashasav10/Walmart-Recruiting-II-Sales-in-Stormy-Weather/master/Data/train_logistic.csv',usecols=['tmax', 'tmin','tavg','wetbulb','heat','cool','sealevel','station_number','store_nbr','item_nbr','stnpressure','resultspeed','sunrise', 'sunset','depart', 'dewpoint']);
y = pd.read_csv('https://raw.githubusercontent.com/sashasav10/Walmart-Recruiting-II-Sales-in-Stormy-Weather/master/Data/train_logistic.csv',usecols=['units'])
y['units'] = np.log(y.units+1)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
xMatrix = X_train.values
yMatrix = y_train.values
regr = linear_model.LinearRegression()
regr.fit(xMatrix, yMatrix)

x_predict = X_test;
print(x_predict)

y_predict = regr.predict(x_predict)
print(y_predict)

print('Accuracy of Linear Regression - Data set: {:.2f}'.format(regr.score(X_test, y_test)))

# # LINEAR REGRESSION
# ######################
# # define the model
# lr = LinearRegression()
#
# # build the pipeline
# lr_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("lr", lr)])
#
# # fit the model
# lr_pipeline.fit(X_train, y_train)
#
# # get prediction
# lr_pred = lr_pipeline.predict(X_test)
#
# # mean absolute error
# lr_MAE = round(mean_absolute_error(y_test, lr_pred), 2)
#
# #  r2 score
# lr_r2 = round(r2_score(y_test, lr_pred), 2)
#
# # hyperparameter tuning
# lr_param_grid = {"lr__fit_intercept": [True, False]}
#
# # define the Gridsearch object
# lr_grid_search = GridSearchCV(lr_pipeline, lr_param_grid, cv=5)
#
# # fit the Gridsearch object
# lr_grid_search.fit(X_train, y_train)
#
# # Cross Validation score
# lr_CV = round(lr_grid_search.best_score_, 2)
#
# # validation set score
# lr_valid_score = round(lr_grid_search.score(X_test, y_test), 2)
# print("Best parameters (LinearRegressor): {} \n".format(lr_grid_search.best_params_))

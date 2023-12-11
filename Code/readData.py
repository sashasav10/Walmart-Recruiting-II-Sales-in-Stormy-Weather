import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
import csv
from sklearn import datasets, linear_model
from sklearn import svm

csv_database = create_engine('sqlite:///csv_database.db')
path = ('https://raw.githubusercontent.com/sashasav10/Walmart-Recruiting-II-Sales-in-Stormy-Weather/master/Data'
        '/train_logistic.csv')
x = pd.read_csv(path+'train3.csv', usecols=['tmax', 'tmin','tavg','wetbulb','heat','cool','sealevel','station_number','store_nbr','item_nbr','stnpressure','resultspeed','sunrise', 'sunset','depart', 'dewpoint', 'wetbulb','snowfall', 'preciptotal'])
y = pd.read_csv(path+'train3.csv', usecols=['units'])
xMatrix = x.as_matrix()
yMatrix = y.as_matrix()
regr = linear_model.LinearRegression()
regr.fit(xMatrix, yMatrix)

x_predict = pd.read_csv(path+'train3.csv', usecols=['tmax', 'tmin','tavg','wetbulb','heat','cool','sealevel','station_number','store_nbr','item_nbr','stnpressure','resultspeed','sunrise', 'sunset','depart', 'dewpoint', 'wetbulb','snowfall', 'preciptotal'])
y_predict = regr.predict(x_predict)
print(y_predict)


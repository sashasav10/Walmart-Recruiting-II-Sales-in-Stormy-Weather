import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
import csv
from sklearn import datasets, linear_model
from sklearn import svm

csv_database = create_engine('sqlite:///csv_database.db')

trainData = pd.read_csv('https://github.com/sashasav10/Walmart-Recruiting-II-Sales-in-Stormy-Weather/blob/master/Data/train.csv')

def createCsv(csv_file_name,csv_columns):
	returnWriter=None
	try:
		with open(csv_file_name, 'w',newline="") as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
			writer.writeheader()
			returnWriter=writer
	except IOError:
		print("Error occurred with")
	return returnWriter

def writeToCsv(writer,data):
	try:
		writer.writerow(data) 
	except IOError:
		print("Error occurred with")
	return
	
def WriteDictToCSV(csv_file,csv_columns,dict_data):
    try:
        with open(csv_file, 'w',newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("Error occurred with")
    return

writer = None;
count = 100000;
trainWeatherData = []
for index, row in trainData.iterrows():
	storeNumber = row['store_nbr']
	dateAry		= row['date'].split("-")
	date = dateAry[2]+"-"+dateAry[1]+"-"+dateAry[0]
	stationNumberResult = pd.read_sql_query('SELECT station_nbr FROM key where store_nbr='+str(storeNumber), csv_database)
	stationNumber = None
	for index,keyRow in stationNumberResult.iterrows():
		stationNumber= keyRow['station_nbr']
	weatherDetails = pd.read_sql_query('SELECT * FROM weather where station_number='+str(stationNumber)+' and datee="'+str(date)+'"', csv_database)
	for index,weatherRow in weatherDetails.iterrows():
		weatherRow['date'] 			= row['date']
		weatherRow['store_nbr'] 	= storeNumber
		weatherRow['item_nbr'] 		= row['item_nbr'] 	
		weatherRow['units']			= row['units']
		row2dict = dict(weatherRow)
		trainWeatherData.append(row2dict);
	count = count - 1
	if count == 0:
		break

x = pd.DataFrame(trainWeatherData,columns=['tmax', 'tmin','tavg','wetbulb','heat','cool','sealevel','station_number','store_nbr','item_nbr','stnpressure','resultspeed','sunrise', 'sunset','depart', 'dewpoint']);
y = pd.DataFrame(trainWeatherData, columns=['units'])
y['units'] = np.log(y.units+1)
xMatrix = x.as_matrix()
yMatrix = y.as_matrix()
regr = linear_model.LinearRegression()
regr.fit(xMatrix, yMatrix)

x_predict = pd.read_csv(path+'train3.csv', usecols=['tmax', 'tmin','tavg','wetbulb','heat','cool','sealevel','station_number','store_nbr','item_nbr','stnpressure','resultspeed','sunrise', 'sunset','depart', 'dewpoint'])
y_predict = regr.predict(x_predict)
print(y_predict)

import os
import sys
sys.stderr = open(os.getcwd() + "\\err.txt", 'w') # devia gli errori nel file

from datetime import datetime, date
from math import sqrt
import numpy as np
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
scaler_filename = "scaler.save"



Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
seasons = [(1, (date(Y,  1,  1),  date(Y,  3, 20))), # winter
			(2, (date(Y,  3, 21),  date(Y,  6, 20))), # spring
			(3, (date(Y,  6, 21),  date(Y,  9, 22))), # summer
			(4, (date(Y,  9, 23),  date(Y, 12, 20))), # fall
			(1, (date(Y, 12, 21),  date(Y, 12, 31)))] # winter

def get_season(now):
	if isinstance(now, datetime):
		now = now.date()
	now = now.replace(year=Y)
	return next(season for season, (start, end) in seasons
				if start <= now <= end)

def num(s):
	try:
		return int(s)
	except ValueError:
		return str(s)
def normalize(val, minim, maxim):
	return((val-minim)/(maxim-minim))

def getWeatherLabel(weather):
	weather = weather.lower()
	if "heavy" in weather or "snow" in weather or "ice" in weather:
		return 4
	elif "rain" in weather or "light" in weather:
		return 3
	elif "cloud" in weather or "mist" in weather:
		return 2
	else:
		return 1

def make_prediction(X):
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	# print("Loaded model from disk")
	# evaluate loaded model on test data
	loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
	# make a prediction
	return loaded_model.predict(X)

# timestamp , weather , temp ,humidity, windspeed
def convert(lista):
	timestamp = num(lista[0])

	if isinstance(timestamp, int):
		date = datetime.fromtimestamp(timestamp)
		srt_datetime = date.strftime('%Y-%m-%d %H:%M:%S')
		season = get_season(date)
		month = date.strftime('%m')
		hour = date.strftime('%H')
		weekday = date.weekday() # 0 = monday, 6 = sunday
		weekday = (weekday + 1) % 7 # weekday fix to 0 = sunday, 6 = saturday
		holiday = 1 if weekday == 0 or weekday == 6 else 0
		workingday = int(not holiday)
		weather = lista[1]
		weather_label = getWeatherLabel(weather)
		norm_temp = round(normalize(float(lista[2]) - 273.15, minim = -8.0, maxim = 39.0), 4)
		humidity = float(lista[3]) / 100
		windspeed = float(lista[4]) * 3.6
		norm_windspeed = round(windspeed / 67, 4)
		# print (season, month, hour, weekday, holiday, workingday, weather_label, norm_temp, humidity, norm_windspeed)
		return [season, month, hour, holiday, weekday, workingday, weather_label, norm_temp, humidity, norm_windspeed]

f = open(sys.argv[1], 'r')
predictions = []
for line in f:
	line =  line.rstrip()
	pcs = line.split(",")
	row = convert(pcs)
	if row is not None: predictions.append(row)
#print(predictions)
scaler = joblib.load(scaler_filename)
X = scaler.transform(np.array(predictions)) #season, month, hour, holiday, weekday, workingday, weather_label, norm_temp, humidity, norm_windspeed
res = make_prediction(X)
for i in res:
	print(int(i))
sys.stdout.flush()

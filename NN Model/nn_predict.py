from math import sqrt
import numpy as np
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
import numpy
import sys

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
# make a prediction
yhat = loaded_model.predict(test_X)
# calculate RMSE
rmse = sqrt(mean_squared_error(yhat, test_y))

print('Test RMSE: %.3f' % rmse)

pyplot.plot(yhat, label='prediction')
pyplot.plot(test_y, label='actual')
pyplot.legend()
pyplot.show()

sys.stdout.flush()

'''
season  | min:  1  - max:  4
yr  | min:  0  - max:  1
mnth  | min:  1  - max:  12
hr  | min:  0  - max:  23
holiday  | min:  0  - max:  1
weekday  | min:  0  - max:  6
workingday  | min:  0  - max:  1
weathersit  | min:  1  - max:  4
temp  | min:  0.02  - max:  1.0
atemp  | min:  0.0  - max:  1.0
hum  | min:  0.0  - max:  1.0
windspeed  | min:  0.0  - max:  0.8507
'''
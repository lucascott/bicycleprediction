from math import sqrt
import numpy as np
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from tensorflow import set_random_seed
import numpy
from sklearn.externals import joblib



# for Reproducibility
np.random.seed(7)
set_random_seed(7)

filename = "hour.csv"
scaler_filename = "scaler.save"
train_percentage = 90
epochs = 100

# load dataset
df = pd.read_csv(filename)


df.drop(["instant","yr","atemp","dteday","casual","registered"], inplace= True, axis = 1)
# ensure that the y columns will be the last of the dataset
cols_at_end = ["cnt"]
df = df[[c for c in df if c not in cols_at_end] + [c for c in cols_at_end if c in df]]
print(df.columns)
print(df.head())


for x in df.columns.values.tolist():
	print (x," | min: ", df[x].min(), " - max: ", df[x].max()," - avg: ", df[x].mean())


# integer encode direction
#encoder = LabelEncoder()
#values[:,4] = encoder.fit_transform(values[:,4])

# ensure all data is float
df.values.astype('float32')

# normalize features
df.interpolate(inplace=True) # rimuovo i valori NaN altrimenti non posso normalizzare
scaler = joblib.load(scaler_filename)
scaled = scaler.transform(df.values[:,: -len(cols_at_end)])
scaled = pd.DataFrame(scaled)
scaled = pd.concat((scaled, df[cols_at_end]), axis=1)
print(scaled.head())

# shuffle the dataset
scaled = scaled.sample(frac=1).reset_index(drop=True)

# split into train and test sets
values = scaled.values
split_index = round(scaled.shape[0]*train_percentage/100)
train = values[:split_index, :]
test = values[split_index:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


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
# make predictions
yhat = loaded_model.predict(test_X)
score = loaded_model.evaluate(test_X, test_y)
# calculate RMSE and absolute error
rmse = sqrt(mean_squared_error(yhat, test_y))
absolute = mean_absolute_error(yhat, test_y)

print("Test accuracy: %.3f" % score[1])
print('Test mean absolute error: %.3f' % absolute)
print('Test RMSE: %.3f' % rmse)

pyplot.plot(yhat, label='prediction')
pyplot.plot(test_y, label='actual')
pyplot.legend()
pyplot.show()


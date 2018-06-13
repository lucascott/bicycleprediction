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
from keras.layers import LSTM
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
epochs = 50

# load dataset
df = pd.read_csv(filename)


df.drop(["instant","yr","atemp","dteday","casual","registered"], inplace= True, axis = 1)


# ensure that the y colomns will be the last of the dataset
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
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(df.values[:,:-1])
scaled = pd.DataFrame(scaled)
scaled = pd.concat((scaled, df[cols_at_end]), axis=1)
joblib.dump(scaler, scaler_filename)
print(scaled.head())

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


# design network
model = Sequential()
model.add(Dense(30, input_dim=train_X.shape[1], activation = "relu"))
model.add(Dense(300, activation = "relu"))
model.add(Dense(120, activation = "relu"))
model.add(Dense(60, activation = "relu"))
#model.add(Dense(units=120, activation = "relu"))
#model.add(Dense(units=60, activation = "relu"))

model.add(Dense(units=1, activation="linear"))

opt = RMSprop()
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
# fit network
history = model.fit(train_X, train_y, epochs=epochs, batch_size=50, validation_data=(test_X, test_y), verbose=2, shuffle=True)
# plot history
pyplot.plot(history.history['loss'], label='loss')
pyplot.plot(history.history['val_loss'], label='val_loss')
pyplot.plot(history.history['acc'], label='accuracy')
pyplot.legend()
pyplot.show()


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



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


"""First working prototype of this model. Achieves an RMSE of ~4.5
after 250 epochs.

Author: Anjo P.
Date: 2/20/19
"""
# TODO: test code on different CDC regions
# TODO: clean code
# TODO: separate preprocessing into a different file

# data wrangling
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error

# machine learning stuff
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import adam

def build_model():
    temp_model = Sequential()
    # return_sequences shouldn't be true but it works
    # with it for some reason
    temp_model.add(LSTM(
        64,
        input_shape=(1, 6),
        return_sequences=False))
    temp_model.add(Dense(1, activation='linear'))

    # compile model
    temp_model.summary()
    temp_model.compile(loss='mean_absolute_error',
                       optimizer='adam', metrics=['accuracy'])
    return temp_model

model = build_model()

agg_df = pd.read_csv("data/TOTALINPUT.csv")
pred_df = pd.read_csv("data/TOTALPRED.csv")

x_arr = np.asarray(agg_df)
x_arr = x_arr.reshape(5615, 1, 6)
print(x_arr)

y_arr = np.asarray(pred_df)
print(y_arr)

split = int(round(0.2 * x_arr.shape[0]))
print("Training data has", split, "rows")
X_train = x_arr[split:, :, :]
Y_train = y_arr[split:, :]
x_test = x_arr[:split, :, :]
y_test = y_arr[:split, ]

print(X_train.shape)
print(Y_train.shape)

# print("\n\n-------------------\nFit Numero Uno\n\n")
model.fit(X_train, Y_train, epochs=400, batch_size=1,
            shuffle=False, validation_data=(x_arr, y_arr))

"""
agg_df = pd.read_csv("data/pacific/AGGDATA.csv")
pred_df = pd.read_csv("data/pacific/PREDDATA.csv")

x_arr = np.asarray(agg_df)
x_arr = x_arr.reshape(624, 1, 6)

y_arr = np.asarray(pred_df)
"""

# predicc
predicted = model.predict(x_test)
predicted = np.reshape(predicted, (predicted.size,))
print("\n")
print(predicted.shape)
print(y_test.shape)
print("\n")

rmse = sqrt(mean_squared_error(y_test, predicted))
print("rmse=", rmse)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_test[:100])
plt.plot(predicted[:100])
plt.show()

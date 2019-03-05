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

# pacific
# mountain
# new england
# middle atlantic
# south atlantic
# east north central
# east south central
# west north central
# west south central
regions = ["pacific", 
            "mountain",
            "new england",
            "middle atlantic",
            "south atlantic",
            "east north central",
            "east south central",
            "west north central",
            "west south central"]


def build_model():
    temp_model = Sequential()
    # return_sequences shouldn't be true but it works
    # with it for some reason
    temp_model.add(LSTM(
        8,
        input_shape=(1, 6),
        return_sequences=False))
    temp_model.add(Dense(1, activation='linear'))

    # compile model
    temp_model.summary()
    temp_model.compile(loss='mean_squared_error',
                       optimizer='adam', metrics=['accuracy'])
    return temp_model

model = build_model()

for region in regions:
    agg_df = pd.read_csv("data/{0}/AGGDATA.csv".format(region))
    pred_df = pd.read_csv("data/{0}/PREDDATA.csv".format(region))

    x_arr = np.asarray(agg_df)
    x_arr = x_arr.reshape(624, 1, 6)
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
    model.fit(X_train, Y_train, epochs=250, batch_size=1,
                shuffle=False, validation_data=(x_arr, y_arr))

# predicc
predicted = model.predict(x_test)
predicted = np.reshape(predicted, (predicted.size,))
print("\n")
print(predicted.shape)
print(y_test.shape)
print("\n")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_test[:100])
plt.plot(predicted[:100])
plt.show()

rmse = sqrt(mean_squared_error(y_test, predicted))
print("rmse=",rmse)

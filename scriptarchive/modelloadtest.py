from keras import backend as K
"""Deprecated. Used incorrectly prepared training data."""
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
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.optimizers import adam
from keras.models import load_model

def build_model():
    temp_model = Sequential()
    # return_sequences shouldn't be true but it works
    # with it for some reason
    temp_model.add(LSTM(
        16,
        input_shape=(1, 6),
        return_sequences=False))
    temp_model.add(Dropout(0.2))
    temp_model.add(Dense(1, activation='linear'))

    # compile model
    temp_model.summary()
    temp_model.compile(loss='mean_squared_error',
                       optimizer='adam', metrics=['mean_squared_error'])
    return temp_model

def reload_model():
    temp_model = load_model("savetest2/test0.h5")
    return temp_model


model = reload_model()

agg_df = pd.read_csv("data/TOTALINPUT.csv")
pred_df = pd.read_csv("data/TOTALPRED.csv")

x_arr = np.asarray(agg_df)
x_arr = x_arr.reshape(5615, 1, 6)
print(x_arr)

y_arr = np.asarray(pred_df)
print(y_arr)

split = int(round(0.8 * x_arr.shape[0]))
print("Training data has", split, "rows")
X_train = x_arr[:split, :, :]
Y_train = y_arr[:split, :]
x_test = x_arr[split:, :, :]
y_test = y_arr[split:, ]

print(X_train.shape)
print(Y_train.shape)

# predicc
predicted = model.predict(x_test)
predicted = np.reshape(predicted, (predicted.size,))
print("\n")
print(predicted.shape)
print(y_test.shape)
print("\n")

# 7.1062751390509735
rmse = sqrt(mean_squared_error(y_test, predicted))
print("preliminary rmse=", rmse)

for i in range(9):
    # print("\n\n-------------------\nFit Numero Uno\n\n")
    model.fit(X_train, Y_train, epochs=100, batch_size=1,
                shuffle=False, validation_data=(x_arr, y_arr))

    # predicc
    predicted = model.predict(x_test)
    predicted = np.reshape(predicted, (predicted.size,))
    print("\n")
    print(predicted.shape)
    print(y_test.shape)
    print("\n")

    rmse = sqrt(mean_squared_error(y_test, predicted))
    print("rmse=", rmse)

    model.save("savetest2/test{0}.h5".format(i))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_test[:250])
    plt.plot(predicted[:250])
    plt.savefig("savetest2/test{0}.png".format(i))
    plt.close()



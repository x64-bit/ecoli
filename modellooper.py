"""Attempt to perform 9-fold crossvalidation. Achieves an RMSE of ~4.5
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# pacific not included for testing
regions = ["pacific",
    "mountain",
    "new england",
    "middle atlantic",
    "south atlantic",
    "east north central",
    "east south central",
    "west north central",
    "west south central"]

epochs = 1

def build_model():
    """Builds a model"""
    temp_model = Sequential()
    # return_sequences shouldn't be true but it works
    # with it for some reason
    temp_model.add(LSTM(
        16,
        input_shape=(1, 6),
        return_sequences=False))
    # avoid overfitting
    temp_model.add(Dropout(0.2))
    temp_model.add(Dense(1, activation='linear'))

    # compile model
    temp_model.summary()
    temp_model.compile(loss='mean_squared_error',
                       optimizer='adam', metrics=['mean_absolute_error'])
    return temp_model

for removed_region in regions:
    print("\n\n\nBeginning new fold...\n-----------------------------\n\n\n")
    print("Removing region", removed_region,
          "\n\n\n-----------------------------")
    temp_regions = regions.copy()
    temp_regions.remove(removed_region)

    print("Building new model...\n\n\n-----------------------------")
    model = build_model()
    
    for region in temp_regions:
        print("Training on regions", temp_regions)
        # read data from specified region
        agg_df = pd.read_csv("data/{0}/AGGDATA.csv".format(region))
        pred_df = pd.read_csv("data/{0}/PREDDATA.csv".format(region))

        # convert to arrays, reshape to fit
        x_arr = np.asarray(agg_df)
        x_arr = x_arr.reshape(676, 1, 6)
        y_arr = np.asarray(pred_df)

        # rows used for training
        train_split = int(round(0.7 * x_arr.shape[0]))
        # last row of validation (first row is row after end of train split)
        val_split = int(round(0.85 * x_arr.shape[0]))
        print("Training data has", train_split, "rows")
        # split train
        X_train = x_arr[:train_split, :, :]
        Y_train = y_arr[:train_split, :]
        # split val
        x_val = x_arr[train_split:val_split, :, :]
        y_val = y_arr[train_split:val_split, ]
        # split test
        x_test = x_arr[val_split:, :, :]
        y_test = y_arr[val_split:, ]

        print(X_train.shape)
        print(Y_train.shape)

        # fit on sample for 250 epochs
        model.fit(X_train, Y_train, epochs=epochs, batch_size=1,
                    shuffle=False, validation_data=(x_arr, y_arr))

        """Create predictions"""
        predicted = model.predict(x_test)
        predicted = np.reshape(predicted, (predicted.size,))
        print("\n")
        print(predicted.shape)
        print(y_test.shape)
        print("\n")

        rmse = sqrt(mean_squared_error(y_test, predicted))
        mae = mean_absolute_error(y_test, predicted)
        print("rmse=", rmse)
        print("mae=", mae)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_test[:100])
        plt.plot(predicted[:100])
        plt.savefig("test.png")
        plt.close()
    
    model.save("models/{0} removed.h5".format(removed_region))

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
from keras.models import load_model

# etc
import os

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

epochs = 200
rmse_array = []
mae_array = []
fold_num = 0

for removed_region in regions:
    """THIS LOOP TRAINS PER FOLD"""
    print("\n\n\nBeginning new fold...\n-----------------------------\n\n\n")
    print("Removing region", removed_region,
          "\n\n\n-----------------------------")
    temp_regions = regions.copy()
    temp_regions.remove(removed_region)

    print("Creating filepath...")
    # create path to save models, metrics
    save_path = "models/200epochs/fold" + str(fold_num) + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Loading model...\n\n\n-----------------------------")
    model = load_model(save_path + "{0} removed.h5".format(removed_region))
    
    for region in temp_regions:
        """THIS LOOP TRAINS PER SAMPLE"""
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

        """Create model predictions"""
        predicted = model.predict(x_test)
        predicted = np.reshape(predicted, (predicted.size,))

        # metrics for this fold
        test_rmse = sqrt(mean_squared_error(y_test, predicted))
        test_mae = mean_absolute_error(y_test, predicted)
        print("test_rmse=", test_rmse)
        print("test_mae=", test_mae)

        rmse_array.append(test_rmse)
        mae_array.append(test_mae)

        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_test)
        plt.plot(predicted)
        plt.title("Model Predictions: " + str(epochs) + " epochs")
        plt.xlabel("Weeks")
        plt.xticks(np.arange(0, len(y_test), 5))
        plt.ylabel("Cases of E. coli")
        plt.yticks(np.arange(0, 101, 5))
        plt.savefig(
            save_path + "{0} removed predictions.png".format(region))
        plt.close()
    
    fold_num+=1

result_rmse = np.mean(rmse_array)
result_mae = np.mean(mae_array)

# read data from specified region
agg_df = pd.read_csv("data/pacific/AGGDATA.csv")
pred_df = pd.read_csv("data/pacific/PREDDATA.csv")

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

"""Create model predictions"""
predicted = model.predict(x_test)
predicted = np.reshape(predicted, (predicted.size,))

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_test)
plt.plot(predicted)
plt.title("Model Predictions: " + str(epochs) + " epochs")
plt.xlabel("Weeks")
plt.xticks(np.arange(0, len(y_test), 5))
plt.ylabel("Cases of E. coli")
plt.yticks(np.arange(0, 101, 5))
plt.savefig(
    "models/" + "{0} removed predictions.png".format(removed_region))
plt.close()

# write scores
with open("C:/Users/Walter/Documents/GitHub/ecoli/models/scores.txt", 'w+') as scores:
    scores.write("rmse values:\n")
    for i in rmse_array:
        scores.write(str(i))
        scores.write('\n')
    scores.write("mae values:\n")
    for i in mae_array:
        scores.write(str(i))
        scores.write('\n')
    scores.write('avg. rmse: ' + str(result_rmse) + "\n")
    scores.write('avg. mae: ' + str(result_mae))

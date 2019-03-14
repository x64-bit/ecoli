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

for region in regions:
    agg_df = pd.read_csv("data/{0}/AGGDATA.csv".format(region))
    pred_df = pd.read_csv("data/{0}/PREDDATA.csv".format(region))

    x_arr = np.asarray(agg_df)
    x_arr = x_arr.reshape(624, 1, 6)

    y_arr = np.asarray(pred_df)
    print(pred_df)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_arr)
    plt.show()
    plt.close()

df = pd.read_csv("data/pacific/EPI2018-NNDSS.csv")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(df)
plt.show()
plt.close()

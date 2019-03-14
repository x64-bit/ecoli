"""Deprecated. Used incorrectly prepared training data."""

# data wrangling
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

regions = ["pacific",
           "mountain",
           "new england",
           "middle atlantic",
           "south atlantic",
           "east north central",
           "east south central",
           "west north central",
           "west south central"]

baseline_rmse = []
baseline_mae = []

for region in regions:
    print("Region:", region)
    agg_df = pd.read_csv("data/{0}/AGGDATA.csv".format(region))
    pred_df = pd.read_csv("data/{0}/PREDDATA.csv".format(region))
    naive_df = agg_df.loc[:,"Cases"]

    x_arr = np.asarray(naive_df)
    # print(x_arr)

    y_arr = np.asarray(pred_df)
    # print(y_arr)

    split = int(round(0.8 * x_arr.shape[0]))
    print("Training data has", split, "rows")
    x_test = x_arr[split:, ]
    y_test = y_arr[split:, ]

    rmse = sqrt(mean_squared_error(x_test, y_test))
    mae = mean_absolute_error(x_test, y_test)
    print("rmse=",rmse)
    print("mae=", mae)

    baseline_rmse.append(rmse)
    baseline_mae.append(mae)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_test[:100])
    ax.plot(y_test[:100])
    plt.show()
    plt.close()
    print("--------------------------\n\n\n\n")

finalrmse = np.mean(baseline_rmse)
finalmae = np.mean(baseline_mae)

print(finalrmse)
print(finalmae)
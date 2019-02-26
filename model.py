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

# pytrends
from pytrends.request import TrendReq

# machine learning stuff
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import adam

# initialize dataframes for search queries and epidemic data
trends_df = pd.DataFrame()
epi_df = pd.DataFrame()

# set up pytrends
# langauge is US English, timezone set to california local
pytrends = TrendReq(hl='en-US', tz=480)

# keywords to parse
kw_list = ["symptoms of e coli",
           "coli",
           "signs of e coli",
           "e coli symptoms",
           "e coli"]

# at its core, pytrends is just a webscraper
# if you open multiple years in Google Trends, it will only let you download
# .csv files for the months
# that means we'll have to scrape them from each year separately to get weeks
for i in range(10):
  year = 2006 + i
  
  # build payload for this year
  pytrends.build_payload(kw_list=kw_list, 
                         timeframe='{0}-1-1 {1}-12-31'.format(year,year), 
                         geo='US-CA')
  # scrape data
  temp_df = pytrends.interest_over_time()
  trends_length = len(temp_df.index)
  # counted by calendar weeks, not literal chunks of 7 days
  # just removed results for first day to account for this
  if (trends_length > 52):
    temp_df = temp_df.iloc[1:]
  print("Appending search dataframe of length: ", len(temp_df.index))
  trends_df = pd.concat([trends_df, temp_df])
  
  # append new epidemic data
  temp_df = pd.read_csv("Data/EPI{0}-NNDSS.csv".format(year))
  # for some reason the CDC data duplicated the data for week 48 of 2006
  # I have to account for this
  if (year == 2006):
    temp_df = temp_df.drop([48])
  print("Appending epidemic dataframe of length: ", len(temp_df.index))
  epi_df = pd.concat([epi_df, temp_df])

epi_df = epi_df.reset_index(drop=True)

trends_df.drop(['isPartial'], axis=1, inplace=True)
trends_df.reset_index(drop=True, inplace=True)
print(trends_df)
print(epi_df)

agg_df = pd.DataFrame()

pred_df = epi_df
# concatenate all data and shift one back. pred_df will be predictions
agg_df = pd.concat([trends_df, epi_df], axis=1).shift(-1)

agg_df = agg_df.fillna(0)
print(agg_df)

x_arr = np.asarray(agg_df)
x_arr = x_arr.reshape(520,1,6)
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

def build_model():
    model = Sequential()
    # return_sequences shouldn't be true but it works
    # with it for some reason
    model.add(LSTM(
        8,
        input_shape=(1, 6),
        return_sequences=False))
    model.add(Dense(1, activation='linear'))
    return model


# help
test_model = build_model()
test_model.summary()
test_model.compile(loss='mean_squared_error',
                   optimizer='adam', metrics=['accuracy'])
print("\n\n-------------------\nFit Numero Uno\n\n")
test_model.fit(X_train, Y_train, epochs=250, batch_size=1,
               shuffle=False, validation_data=(x_arr, y_arr))

predicted = test_model.predict(x_test)
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

print("\n\n-------------------\nFit Numero Dos\n\n")
test_model.fit(X_train, Y_train, epochs=250, batch_size=1,
               shuffle=False, validation_data=(x_arr, y_arr))

predicted = test_model.predict(x_test)
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
print("rmse=", rmse)

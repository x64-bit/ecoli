"""
Old jank. Don't use. Only here for reference.
"""


# -*- coding: utf-8 -*-
"""preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d6IzuJzKR6q0YKXLLwwoloOzrx73sID4
"""

'''
Data preprocessing

This LSTM uses 6 inputs:
    - STEC cases from the last week
    - Popularity of five search queries
        - "coli"
        - "e coli symptoms"
        - "e coli"
        - "signs of e coli"
        - "symptoms of e coli"

The goal of this project is to get the network to predict the first input,
then plug it into itself recursively to make weekly predictions.

All of these data are stored in .csv files though, so we have to
parse through them before we can do anything useful.

This is my first time doing preprocessing so the code here is
absolutely jank. Don't use this as a reference lol
'''

# data handling
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

# machine learning stuf
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

# misc
import time

# seed for reproducibility
np.random.seed(1234)

# weekly search query data. the actual query is in the filename immediately
# after "SEARCH2018-"
trends_1 = pd.read_csv("Data/SEARCH2017-coli.csv")
trends_2 = pd.read_csv("Data/SEARCH2017-e coli symptoms.csv")
trends_3 = pd.read_csv("Data/SEARCH2017-e coli.csv")
trends_4 = pd.read_csv("Data/SEARCH2017-signs of e coli.csv")
trends_5 = pd.read_csv("Data/SEARCH2017-symptoms of e coli.csv")
print(trends_1)

# NNDSS data; note this is all from 2018
# the columns we extracted were the region, week, and # of cases
epi_raw = pd.read_csv("Data/EPI2018-NNDSS.csv",
                      index_col=" Reporting Area", usecols=[0, 2, 13])

# this took FAR longer than it should have because I skim everything
# epi_raw.iterrows() generates a list of tuples with (row index, entire row attributes)
#                                                       ^ in this case, row index = index_col above
# it loops through the entire DataFrame, checking if the index is "PACFIC".
# if it is, keep it, otherwise discard the row.
# the end result should be a dataframe with week-by-week data on e. coli infections.
# later on I might edit this so that the index is the week, and I only check for
# an element of a row.
for index, row in epi_raw.iterrows():
    print(index)
    if index != "PACIFIC":
        epi_raw.drop(index, inplace=True)
        print("deleted!")
    else:
        print("\tcorrect!")
        
# handle missing data, which means there wasn't any recorded cases
epi_raw = epi_raw.fillna(0)
print(epi_raw)

# convert case numbers to numpy array, then pandas series for concatenation
pred_case_array = pd.Series(
    np.asarray(
        epi_raw["Shiga toxin-producing E. coli (STEC)§, Current week"])).fillna(0)
# case_array but shifted 1 week ahead for predictions
train_case_array = pred_case_array.shift(-1).fillna(0)
print(train_case_array)

# convert trend data to numpy array, then pandas series for concat
# for some reason, the column header was included as an item so I had to leave
# that out of the list
trends_array1 = pd.Series(np.asarray(trends_1.iloc[1:53, 0])).shift(-1).fillna(0)
trends_array2 = pd.Series(np.asarray(trends_2.iloc[1:53, 0])).shift(-1).fillna(0)
trends_array3 = pd.Series(np.asarray(trends_3.iloc[1:53, 0])).shift(-1).fillna(0)
trends_array4 = pd.Series(np.asarray(trends_4.iloc[1:53, 0])).shift(-1).fillna(0)
trends_array5 = pd.Series(np.asarray(trends_5.iloc[1:53, 0])).shift(-1).fillna(0)
print(trends_array2)

# concatenate all data into one giant dataframe for easy usage
agg_df = pd.concat([trends_array1, trends_array2, trends_array3, trends_array4, 
                 trends_array5, train_case_array, pred_case_array], axis=1)
print(agg_df)
# column names for clarity. t means the current week
agg_df.columns = ["'coli'(t-1)", 
                  "'e coli symptoms'(t-1)", 
                  "'e coli'(t-1)", 
                  "'signs of e coli'(t-1)", 
                  "'symptoms of e coli'(t-1)", 
                  "cases(t-1)",
                  "cases(t)"]
print(agg_df)

# turn df into numpy array
agg_arr = np.asarray(agg_df).astype(np.int)
print(agg_arr)
# I did it manually then realized I could just call np.asarray.
# The latter was more pythonic.
'''agg_arr = list()
for index, row in agg_df.iterrows():
  temp_arr = list()
  for i in range(7):
    temp_arr.append(row[i])
  agg_arr.append(temp_arr)
agg_arr = np.asarray(agg_arr)
print(agg_arr)'''

# reshape agg_arr to (1, 52, 7) - 1 sample, being the whole year, 52 time steps,
#                                 and 7 features per time step
agg_arr = agg_arr.reshape(1,52,7)
print(agg_arr.shape)

length = agg_arr.shape[0]
train_length = int(length * .8)
test_length = length - train_length
# broken fix tmrw
'''
train_x = agg_arr[0:train_length][0:5].reshape(1, 52, 6)
train_y = agg_arr[0:train_length][:6].reshape(1, 52, 1)
test_x = agg_arr[0:train_length][0:5].reshape(1, 52, 6)
train_y = agg_arr[0:train_length][:6].reshape(1, 52, 1)
'''

print(train_x)
print("shape: ", train_x.shape)

# no idea if I processed the data right but here goes
def build_model():
    model = Sequential()
    model.add(LSTM(6, input_shape=(1,52,7), return_sequences=True))
    model.add(Dense(1,activation='tanh'))
    return model

test_model = build_model()
test_model.summary()
test_model.compile(loss='mean_squared_error', optimizer='adam')
test_model.fit(train_x,train_y, epochs=1, batch_size=1, shuffle=False)


"""Compiles files into easy to read training and testing csv.

I know I could do that in the same file as the other ones, just feel like it's
neater here.
"""
# TODO
# pacific
# mountain
# new england 
# middle atlantic
# south atlantic
# east north central
# east south central
# west north central
# west south central

# data wrangling
import pandas as pd
import csv
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

# just in case
import matplotlib.pyplot as plt

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
  print("computing", region)
  # will parse instead
  trends_df = pd.read_csv("data/{0}/SEARCH20XX.csv".format(region))

  # set up epidemic case dataframe
  epi_df = pd.DataFrame()
  # print(epi_df)

  # at its core, pytrends is just a webscraper
  # if you open multiple years in Google Trends, it will only let you download
  # .csv files for the months
  # that means we'll have to scrape them from each year separately to get weeks
  for i in range(13):
    year = 2006 + i
    
    # append new epidemic data
    temp_df = pd.read_csv("data/{0}/EPI{1}-NNDSS.csv".format(region,year))
    # for some reason the CDC data duplicated the data for week 48 of 2006
    # I have to account for this
    if (year == 2006):
      temp_df = temp_df.drop([48])
    print("Appending epidemic dataframe of length: ", len(temp_df.index))
    epi_df = pd.concat([epi_df, temp_df])

  epi_df = epi_df.reset_index(drop=True)
  print(trends_df)
  print(epi_df)

  agg_df = pd.DataFrame()

  # keep copy of epi_df for predictions
  pred_df = epi_df
  # concatenate all data and shift one back. pred_df will be predictions
  agg_df = pd.concat([trends_df.shift(-1), epi_df.shift(-1)], axis=1)
  # fill missing values
  agg_df = agg_df.fillna(0)

  agg_df.to_csv("data/{0}/AGGDATA.csv".format(region),
                encoding='utf-8', index=False)
  pred_df.to_csv("data/{0}/PREDDATA.csv".format(region),
                encoding='utf-8', index=False)

"""Below was moved to the model definition script
It's kept here for archival purposes.
"""
# x_arr = np.asarray(agg_df)
# x_arr = x_arr.reshape(624,1,6)

# y_arr = np.asarray(pred_df)

# rmse = sqrt(mean_squared_error(np.asarray(epi_df.shift(-1).fillna(0)), np.asarray(pred_df.fillna(0))))
# print(rmse)

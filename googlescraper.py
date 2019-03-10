"""Scrapes keywords for years 2006-2018 with pytrends"""

# data wrangling
import pandas as pd
import csv
import numpy as np

# just in case
import matplotlib.pyplot as plt

# load files in colab
import io

# pytrends
from pytrends.request import TrendReq

from sklearn.metrics import mean_squared_error
from math import sqrt

# will parse instead
trends_df = pd.DataFrame()
# set up epidemic case dataframe
epi_df = pd.DataFrame()

# set up pytrends
pytrends = TrendReq(hl='en-US', tz=480)
# keywords to parse
kw_list = ["symptoms of e coli",
           "coli",
           "signs of e coli",
           "e coli symptoms",
           "e coli"
          ]

# region
# pacific [x]
# mountain [x]
# new england [x]
# middle atlantic [x]
# south atlantic [x]
# east north central [x]
# east south central [x]
# west north central [x]
# west south central [x]
region = "west south central"
# states to parse
states = ["AK", "LA", "OK", "TX", "LA"]

for state in states:
  print("Processing", state + "...")
  # dataframe which will hold data for the year
  year_df = pd.DataFrame()
  # at its core, pytrends is just a webscraper
  # if you open multiple years in Google Trends, it will only let you download
  # .csv files for the months
  # that means we'll have to scrape them from each year separately to get weeks
  for i in range(12):
    year = 2006 + i

    # build payload for this year
    pytrends.build_payload(kw_list=kw_list, 
                           timeframe='{0}-1-1 {1}-12-31'.format(year,year), 
                           geo='US-{0}'.format(state))
    # scrape data
    temp_df = pytrends.interest_over_time()
    trends_length = len(temp_df.index)
    # counted by calendar weeks, not literal chunks of 7 days
    # just removed results for first day to account for this
    if (trends_length > 52):
      temp_df = temp_df.iloc[1:]
    print("Appending search dataframe of length: ", len(temp_df.index))
    year_df = pd.concat([year_df, temp_df])
  
  trends_df = trends_df.add(year_df, fill_value=0)

# get rid of isPartial (not sure what that's for but it's not necessary)
trends_df.drop(['isPartial'], axis=1, inplace=True)
# reset index so we can properly write it to csv
trends_df.reset_index(drop=True, inplace=True)

# find average
trends_df = trends_df.div(len(states))

# write to csv
trends_df.to_csv("data/{0}/SEARCH20XX.csv".format(region), encoding='utf-8', index=False)


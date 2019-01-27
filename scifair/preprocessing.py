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
'''
import pandas as pd
import csv

# search query data. text after "SEARCH2018-" is the query
trends_1 = pd.read_csv("data/SEARCH2018-coli.csv")
trends_2 = pd.read_csv("data/SEARCH2018-e coli symptoms.csv")
trends_3 = pd.read_csv("data/SEARCH2018-e coli.csv")
trends_4 = pd.read_csv("data/SEARCH2018-signs of e coli.csv")
trends_5 = pd.read_csv("data/SEARCH2018-symptoms of e coli.csv")

# NNDSS data; note this is all from 2018
epi_raw = pd.read_csv("data/EPI2018-NNDSS.csv", index_col=0, usecols=[0,2,13])

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
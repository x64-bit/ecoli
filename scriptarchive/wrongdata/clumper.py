"""I did preprocessing wrong for k-fold cross validation, so I'm redoing it here"""

# data wrangling
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error

regions = ["pacific",
           "mountain",
           "new england",
           "middle atlantic",
           "south atlantic",
           "east north central",
           "east south central",
           "west north central",
           "west south central"]

totalinput_df = pd.DataFrame()
totalpred_df = pd.DataFrame()

for region in regions:
    agg_df = pd.read_csv("data/{0}/AGGDATA.csv".format(region))
    pred_df = pd.read_csv("data/{0}/PREDDATA.csv".format(region))

    totalinput_df = pd.concat([totalinput_df, agg_df])
    totalpred_df = pd.concat([totalpred_df, pred_df])

totalinput_df= totalinput_df.fillna(0)
totalpred_df = totalpred_df.fillna(0)

print("Old lengths (input, output): ")
print(len(totalinput_df))
print(len(totalpred_df))

totalinput_df = totalinput_df[:-1]
totalpred_df = totalpred_df[:-1]

print("\nNew lengths (input, output): ")
print(len(totalinput_df))
print(len(totalpred_df))

totalinput_df.to_csv("data/TOTALINPUT.csv", encoding='utf-8', index=False)
totalpred_df.to_csv("data/TOTALPRED.csv", encoding='utf-8', index=False)

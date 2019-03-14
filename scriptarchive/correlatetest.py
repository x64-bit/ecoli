"""Contains several methods for scraping CDC NNDSS data.

Author: Anjo P.
Date: 2/20/19
"""

from tqdm import tqdm
from bs4 import BeautifulSoup
from time import sleep
import requests
import json
import pandas as pd
import time
import os


def gen_year(year, table_name, region, col):
  """Generates Pandas DataFrame with a year's worth of data.
  
  Keyword args:
  year (str) - year of data to pull
  table_name (str) - name of table to pull
  region (str) - region to pull from table
  col (int) - column to pull from table
  """
  filepath = "Data/{0}".format(region)

  # Placeholder dataframe
  df = pd.DataFrame()
  print("Processing year: " + year + "...")
  if (not os.path.exists(filepath)):
    os.mkdir(filepath)

  for i in range(52):
    # Week num
    week = str(i+1)
    if (len(week) < 2):
      week = "0" + week
    print("Processing week", week)

    # Placeholder
    url = (
        "https://wonder.cdc.gov/nndss/static/{0}/{1}/{2}-{3}-table{4}.html"
        .format(year, week, year, week, table_name))
    # Request url
    r = requests.get(url)
    # Load html content into BeautifulSoup for easy parsing
    soup = BeautifulSoup(r.content, 'html.parser')

    # Look for the table body (tbody), then extract all table rows (tr)
    # breaks with 2017 because NNDSS errata is formatted as table
    if (year == "2017" and week == "21"):
      print("broken row, parsing...")
      tables = soup.find_all('table')
      case_table = tables[1]
      rows = case_table.select('tr')
    else:
      rows = soup.select('table tr')
    for row in rows:
      # Extract header, convert to text, eliminate whitespace, use lowercase
      if (row.select_one('th').text.strip().lower() == region.lower()):
        print("Row found!")
        # Extract table values from row
        tds = row.select('td')
        # Find specified col
        cases = tds[col].text.strip()
        print("Cases: [" + cases + "]")
        # Check if col is empty (filled w/ dash mark), if not append to df
        if (cases != "-"):
          df = df.append([int(cases)], ignore_index=True)
        else:
          df = df.append([0], ignore_index=True)

    print("Finished week: " + week)
    print(df)
    df.to_csv("Data/{0}/EPI{1}-NNDSS.csv".format(region, year),
              encoding='utf-8')
    time.sleep(4)
  
  df.columns = ['Cases']
  df.to_csv("Data/{0}/EPI{1}-NNDSS.csv".format(region, year), encoding='utf-8', index=False)
  # f.close()
  return df


def gen_year_early(year, table_name, region, col):
  """Generates Pandas DataFrame with a year's worth of data for CDC MMWR tables. (-2016)
  
  Keyword args:
  year (str) - year of data to pull
  table_name (str) - name of table to pull
  region (str) - region to pull from table
  col (int) - column to pull from table
  """
  # Placeholder dataframe
  df = pd.DataFrame()
  print("Processing year: " + year + "...")
  filepath = "Data/{0}".format(region)
  if (not os.path.exists(filepath)):
    os.mkdir(filepath)

  for i in range(52):
    # Week num
    week = str(i+1)
    if (len(week) < 2):
      week = "0" + week

    print("Processing", week + "...")
    # Placeholder
    url = (
        "https://wonder.cdc.gov/nndss/nndss_weekly_tables_1995_2014.asp?mmwr_year={0}&mmwr_week={1}&mmwr_table={2}&request=Submit"
        .format(year, week, table_name))
    # Request url
    r = requests.get(url)
    # Load html content into BeautifulSoup for easy parsing
    soup = BeautifulSoup(r.content, 'html.parser')

    # Look for the table body (tbody), then extract all table rows (tr)
    for table in soup.find_all('table'):
      for subtable in table.find_all('table'):
        rows = subtable.select('tr')

    for row in rows:
      tds = row.select('td')
      if (tds[0].text.strip().lower() == region.lower()):
        print("Row found!")
        # Find specified col
        cases = tds[col].text.strip()
        print("Cases: [" + cases + "]")
        # Check if col is empty (filled w/ dash mark), if not append to df
        if (cases != "-"):
          df = df.append([int(cases)], ignore_index=True)
        else:
          df = df.append([0], ignore_index=True)

    print("\tFinished week: " + week)
    print(df)
    df.to_csv("Data/{0}/EPI{1}-NNDSS.csv".format(region, year), encoding='utf-8')
    time.sleep(4)

  df.columns = ['Cases']
  df.to_csv("Data/{0}/EPI{1}-NNDSS.csv".format(region, year),
            encoding='utf-8', index=False)
  # f.close()
  return df

"""
# Scraping for Pacific
df_2006 = gen_year_early("2006", "2G", "pacific", 1)
df_2007 = gen_year_early("2007", "2F", "pacific", 6)
df_2008 = gen_year_early("2008", "2F", "pacific", 6)
df_2009 = gen_year_early("2009", "2F", "pacific", 6)
df_2010 = gen_year_early("2010", "2H", "pacific", 6)
df_2011 = gen_year_early("2011", "2H", "pacific", 11)
df_2012 = gen_year_early("2012", "2H", "pacific", 11)
df_2013 = gen_year_early("2013", "2H", "pacific", 11)
df_2014 = gen_year_early("2014", "2I", "pacific", 6)
df_2015 = gen_year_early("2015", "2K", "pacific", 1)
df_2016 = gen_year_early("2016", "2K", "pacific", 1)
df_2017 = gen_year("2017", "2M", "pacific", 6)
df_2018 = gen_year("2018", "2O", "pacific", 6)
# Scraping for Mountain
df_2006 = gen_year_early("2006", "2G", "mountain", 1)
df_2007 = gen_year_early("2007", "2F", "mountain", 6)
df_2008 = gen_year_early("2008", "2F", "mountain", 6)
df_2009 = gen_year_early("2009", "2F", "mountain", 6)
df_2010 = gen_year_early("2010", "2H", "mountain", 6)
df_2011 = gen_year_early("2011", "2H", "mountain", 11)
df_2012 = gen_year_early("2012", "2H", "mountain", 11)
df_2013 = gen_year_early("2013", "2H", "mountain", 11)
df_2014 = gen_year_early("2014", "2I", "mountain", 6)
df_2015 = gen_year_early("2015", "2K", "mountain", 1)
df_2016 = gen_year_early("2016", "2K", "mountain", 1)
df_2017 = gen_year("2017", "2M", "mountain", 6)
df_2018 = gen_year("2018", "2O", "mountain", 6)
time.sleep(30)
# Scraping for New England
df_2006 = gen_year_early("2006", "2G", "new england", 1)
df_2007 = gen_year_early("2007", "2F", "new england", 6)
df_2008 = gen_year_early("2008", "2F", "new england", 6)
df_2009 = gen_year_early("2009", "2F", "new england", 6)
df_2010 = gen_year_early("2010", "2H", "new england", 6)
df_2011 = gen_year_early("2011", "2H", "new england", 11)
df_2012 = gen_year_early("2012", "2H", "new england", 11)
df_2013 = gen_year_early("2013", "2H", "new england", 11)
"""
df_2014 = gen_year_early("2014", "2I", "new england", 6)
df_2015 = gen_year_early("2015", "2K", "new england", 1)
df_2016 = gen_year_early("2016", "2K", "new england", 1)
df_2017 = gen_year("2017", "2M", "new england", 6)
df_2018 = gen_year("2018", "2O", "new england", 6)
time.sleep(30)
# Scraping for Mid. Atlantic
df_2006 = gen_year_early("2006", "2G", "mid. atlantic", 1)
df_2007 = gen_year_early("2007", "2F", "mid. atlantic", 6)
df_2008 = gen_year_early("2008", "2F", "mid. atlantic", 6)
df_2009 = gen_year_early("2009", "2F", "mid. atlantic", 6)
df_2010 = gen_year_early("2010", "2H", "mid. atlantic", 6)
df_2011 = gen_year_early("2011", "2H", "mid. atlantic", 11)
df_2012 = gen_year_early("2012", "2H", "mid. atlantic", 11)
df_2013 = gen_year_early("2013", "2H", "mid. atlantic", 11)
df_2014 = gen_year_early("2014", "2I", "mid. atlantic", 6)
df_2015 = gen_year_early("2015", "2K", "mid. atlantic", 1)
df_2016 = gen_year_early("2016", "2K", "mid. atlantic", 1)
df_2017 = gen_year("2017", "2M", "mid. atlantic", 6)
df_2018 = gen_year("2018", "2O", "middle atlantic", 6)
time.sleep(30)
# Scraping for South Atlantic
df_2006 = gen_year_early("2006", "2G", "s. atlantic", 1)
df_2007 = gen_year_early("2007", "2F", "s. atlantic", 6)
df_2008 = gen_year_early("2008", "2F", "s. atlantic", 6)
df_2009 = gen_year_early("2009", "2F", "s. atlantic", 6)
df_2010 = gen_year_early("2010", "2H", "s. atlantic", 6)
df_2011 = gen_year_early("2011", "2H", "s. atlantic", 11)
df_2012 = gen_year_early("2012", "2H", "s. atlantic", 11)
df_2013 = gen_year_early("2013", "2H", "s. atlantic", 11)
df_2014 = gen_year_early("2014", "2I", "s. atlantic", 6)
df_2015 = gen_year_early("2015", "2K", "s. atlantic", 1)
df_2016 = gen_year_early("2016", "2K", "s. atlantic", 1)
df_2017 = gen_year("2017", "2M", "s. atlantic", 6)
df_2018 = gen_year("2018", "2O", "south atlantic", 6)
time.sleep(30)
# Scraping E.N Central
df_2006 = gen_year_early("2006", "2G", "E.N. CENTRAL", 1)
df_2007 = gen_year_early("2007", "2F", "E.N. CENTRAL", 6)
df_2008 = gen_year_early("2008", "2F", "E.N. CENTRAL", 6)
df_2009 = gen_year_early("2009", "2F", "E.N. CENTRAL", 6)
df_2010 = gen_year_early("2010", "2H", "E.N. CENTRAL", 6)
df_2011 = gen_year_early("2011", "2H", "E.N. CENTRAL", 11)
df_2012 = gen_year_early("2012", "2H", "E.N. CENTRAL", 11)
df_2013 = gen_year_early("2013", "2H", "E.N. CENTRAL", 11)
df_2014 = gen_year_early("2014", "2I", "E.N. CENTRAL", 6)
df_2015 = gen_year_early("2015", "2K", "E.N. CENTRAL", 1)
df_2016 = gen_year_early("2016", "2K", "E.N. CENTRAL", 1)
df_2017 = gen_year("2017", "2M", "E.N. CENTRAL", 6)
df_2018 = gen_year("2018", "2O", "East North Central", 6)
time.sleep(30)
# Scraping W.N Central
df_2006 = gen_year_early("2006", "2G", "W.N. CENTRAL", 1)
df_2007 = gen_year_early("2007", "2F", "W.N. CENTRAL", 6)
df_2008 = gen_year_early("2008", "2F", "W.N. CENTRAL", 6)
df_2009 = gen_year_early("2009", "2F", "W.N. CENTRAL", 6)
df_2010 = gen_year_early("2010", "2H", "W.N. CENTRAL", 6)
df_2011 = gen_year_early("2011", "2H", "W.N. CENTRAL", 11)
df_2012 = gen_year_early("2012", "2H", "W.N. CENTRAL", 11)
df_2013 = gen_year_early("2013", "2H", "W.N. CENTRAL", 11)
df_2014 = gen_year_early("2014", "2I", "W.N. CENTRAL", 6)
df_2015 = gen_year_early("2015", "2K", "W.N. CENTRAL", 1)
df_2016 = gen_year_early("2016", "2K", "W.N. CENTRAL", 1)
df_2017 = gen_year("2017", "2M", "W.N. CENTRAL", 6)
df_2018 = gen_year("2018", "2O", "West North Central", 6)
time.sleep(30)
# Scraping W.S Central
df_2006 = gen_year_early("2006", "2G", "W.S. CENTRAL", 1)
df_2007 = gen_year_early("2007", "2F", "W.S. CENTRAL", 6)
df_2008 = gen_year_early("2008", "2F", "W.S. CENTRAL", 6)
df_2009 = gen_year_early("2009", "2F", "W.S. CENTRAL", 6)
df_2010 = gen_year_early("2010", "2H", "W.S. CENTRAL", 6)
df_2011 = gen_year_early("2011", "2H", "W.S. CENTRAL", 11)
df_2012 = gen_year_early("2012", "2H", "W.S. CENTRAL", 11)
df_2013 = gen_year_early("2013", "2H", "W.S. CENTRAL", 11)
df_2014 = gen_year_early("2014", "2I", "W.S. CENTRAL", 6)
df_2015 = gen_year_early("2015", "2K", "W.S. CENTRAL", 1)
df_2016 = gen_year_early("2016", "2K", "W.S. CENTRAL", 1)
df_2017 = gen_year("2017", "2M", "W.S. CENTRAL", 6)
df_2018 = gen_year("2018", "2O", "West South Central", 6)
time.sleep(30)
# Scraping W.S Central
df_2006 = gen_year_early("2006", "2G", "E.S. CENTRAL", 1)
df_2007 = gen_year_early("2007", "2F", "E.S. CENTRAL", 6)
df_2008 = gen_year_early("2008", "2F", "E.S. CENTRAL", 6)
df_2009 = gen_year_early("2009", "2F", "E.S. CENTRAL", 6)
df_2010 = gen_year_early("2010", "2H", "E.S. CENTRAL", 6)
df_2011 = gen_year_early("2011", "2H", "E.S. CENTRAL", 11)
df_2012 = gen_year_early("2012", "2H", "E.S. CENTRAL", 11)
df_2013 = gen_year_early("2013", "2H", "E.S. CENTRAL", 11)
df_2014 = gen_year_early("2014", "2I", "E.S. CENTRAL", 6)
df_2015 = gen_year_early("2015", "2K", "E.S. CENTRAL", 1)
df_2016 = gen_year_early("2016", "2K", "E.S. CENTRAL", 1)
df_2017 = gen_year("2017", "2M", "E.S. CENTRAL", 6)
df_2018 = gen_year("2018", "2O", "East South Central", 6)
time.sleep(30)

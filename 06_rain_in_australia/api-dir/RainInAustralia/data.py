import pandas as pd
import calendar
import os

from RainInAustralia.parameters import *


train_set = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/raw_data/weatherAUS.csv"


def get_data():
    ''' returns a rain in australia DataFrame '''

    return pd.read_csv(train_set)


def clean_data(df, reduced=False):
    ''' Cleans dataset'''

    # removes duplicates if any
    df = df.drop_duplicates()

    # Removing features that have more than 30% missing values
    df = df.drop(["Sunshine", "Evaporation", "Cloud3pm", "Cloud9am"], axis=1)

    # Removing features that have more than 30% missing values
    df["Location"] = df["Location"].apply(lambda x: badly_named[x] if x in badly_named.keys() else x)

    # Converting to datetime / creating month feature
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].apply(lambda x: x.month)
    df.drop(["Date"], axis=1).reset_index(drop=True)
    df["Month"] = df["Month"].apply(lambda x: calendar.month_name[x])

    # Removes NaN in RainToday and RainTomorrow features and converting them to int
    df = df[(~df["RainTomorrow"].isnull())]
    df = df[(~df["RainToday"].isnull())]
    df["RainTomorrow"] = df["RainTomorrow"].apply(lambda x: 1 if x == "Yes" else 0)
    df["RainToday"] = df["RainToday"].apply(lambda x: 1 if x == "Yes" else 0)

    if reduced is True:
        df = df[["Humidity3pm", "WindGustSpeed", "Location", "Pressure9am", "MinTemp", "RainTomorrow"]]
    return df


if __name__ == '__main__':
    df = get_data()

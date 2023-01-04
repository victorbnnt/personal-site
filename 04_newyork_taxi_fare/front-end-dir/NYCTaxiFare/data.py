import pandas as pd
import os

train_set = "s3://wagon-public-datasets/taxi-fare-train.csv"
test_set = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/raw_data/test.csv"


def get_data(nrows=10_000, predict=False):
    '''returns a DataFrame with nrows on train set from s3 bucket and
       all rows on test set from local file '''

    if predict is True:
        return None, pd.read_csv(test_set)
    else:
        return pd.read_csv(train_set, nrows=nrows), None


def clean_data(df, test=False, predict=False):
    df = df.drop(["key"], axis=1)
    if predict is False:
        df = df.dropna(how='any', axis='rows')
        df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
        df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
        if "fare_amount" in list(df):
            df = df[df.fare_amount.between(0, 4000)]
        df = df[df.passenger_count < 8]
        df = df[df.passenger_count >= 0]
        df = df[df["pickup_latitude"].between(left=40, right=42)]
        df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
        df = df[df["dropoff_latitude"].between(left=40, right=42)]
        df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


if __name__ == '__main__':
    df = get_data()

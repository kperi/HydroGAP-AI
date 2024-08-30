import pandas as pd
import tqdm
import glob
import os
import fire
import joblib
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from feature_utils import split_contiguous_blocks

LAGS = [1, 2, 3]

def create_features(df, lags=[1, 2, 3], dropna=True, use_dummies=True):
    """
    Create lagged features for time series data.

    Parameters:
    - df (DataFrame): The input DataFrame containing the time series data.
    - lags (list): A list of integers representing the lag periods for creating features. Default is [1, 2, 3].
    - dropna (bool): Whether to drop rows with missing values after creating features. Default is True.

    Returns:
    - df_features (DataFrame): The DataFrame with lagged features created from the input data.

    """

    df_features = df.copy()
    for lag in lags:
        df_features[f"P_lag_time_{lag}"] = df_features["tp"].shift(periods=lag)

    for lag in lags:
        df_features[f"Q_lag_time_{lag}"] = df_features["obsdis"].shift(periods=lag)

    P_cols = [f"P_lag_time_{idx}" for idx in range(1, len(lags) + 1)]
    Q_cols = [f"Q_lag_time_{idx}" for idx in range(1, len(lags) + 1)]

    all_cols = ["tp", "obsdis"] + P_cols + Q_cols
    df_features = df_features[all_cols]



    if use_dummies:
        df_features['month'] = df_features.index.month
        df_features['quarter'] = df_features.index.quarter
        df_features = pd.get_dummies(df_features, columns=["month", "quarter"])

    if dropna:
        df_features.dropna(inplace=True)

    return df_features


def create_train_test_splits(df_features, split_ratio=0.8):
    """
    Split the input dataframe into training and testing sets for machine learning modeling.

    Parameters:
    - df_features (pandas.DataFrame): The input dataframe containing the features and target variable.
    - split_ratio (float, optional): The ratio of the dataset to be used for training. Default is 0.8.

    Returns:
    - x_train (pandas.DataFrame): The training features dataframe.
    - y_train (pandas.Series): The training target variable series.
    - x_test (pandas.DataFrame): The testing features dataframe.
    - y_test (pandas.Series): The testing target variable series.
    """

    #df_features = create_features(df_features, lags=LAGS, dropna=True)

    # we keep the fist 80% for training and the last 20% for testing
    # we may want to use a random train/test split
    split_index = int(split_ratio * df_features.shape[0])

    trainset = df_features[:split_index]
    testset = df_features[split_index:]

    x_train = trainset.drop(columns=["obsdis", "tp"])
    y_train = trainset["obsdis"]

    x_test = testset.drop(columns=["obsdis", "tp"])
    y_test = testset["obsdis"]

    return x_train, y_train, x_test, y_test


def create_model(model_type, **argv):
    """
    keep it simple for now: check model name and create corresponding object
    """
    if model_type == "RandomForestRegressor":
        return RandomForestRegressor(**argv)
    elif model_type == "GradientBoostingRegressor":
        return GradientBoostingRegressor(**argv)
    else:
        raise ValueError("Model not implemented")


def train_station_model(
    station_file, model_type="RandomForestRegressor", split_ratio=0.8
):

    df = pd.read_parquet(station_file)

    # split the data frame in blocks of time-contiguous obsdis values
    contiguous_bocks = split_contiguous_blocks(df)


    # create features per block and concatenate all blocks
    df_features = pd.concat(
        [create_features(block, lags=LAGS) for block in contiguous_bocks]
    )

    x_train, y_train, x_test, y_test = create_train_test_splits(
        df_features, split_ratio=split_ratio
    )

    model = create_model(model_type=model_type, n_estimators=100, random_state=42)

    station = os.path.basename(station_file).split("_cleaned")[0].strip(".0")
    if x_train.shape[0] == 0:
        print(f"File for station {station} has no data")
        raise ValueError("No data for station {station}")

    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    return model, score, station


def main(
    data_path: str = "data/parquet.raw",
    model_type="RandomForestRegressor",
    station_name: str = None,
    all_stations: bool = False,
    persist_model: bool = False,
):

    files = glob.glob(data_path + "/*.parquet")

    if all_stations:
        scores = []
        for file in tqdm.tqdm(files):
            _, score, station = train_station_model(file, model_type)
            scores.append((station, score))

        scores = pd.DataFrame(scores, columns=["station", "score"])
        scores = scores.sort_values(by="score", ascending=True)
        print(tabulate(scores.head(), headers="keys", tablefmt="fancy_outline"))
        print(tabulate(scores.tail(), headers="keys", tablefmt="fancy_outline"))

        scores.to_csv(f"scores_{model_type}.csv", index=False)

    elif station_name is not None:
        # train model for a single station
        file = f"{station_name}.0.parquet"
        model, score, station = train_station_model(file, model_type)

        print(f"Score for station {station_name} : {score:0.4}")
        if persist_model:
            model_name = f"{station_name}_{model_type}.joblib"
            joblib.dump(model, model_name)
    else:
        print("Please provide a station id")


if __name__ == "__main__":

    fire.Fire(main)

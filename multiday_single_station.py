import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

import tqdm
import glob
import os
import fire
from tabulate import tabulate

NUM_TREES = 100
DAYS_AHEAD = 14


def create_features(df, lags=[1, 2, 3], dropna=True):
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

    for i in range(DAYS_AHEAD - 1):  # 5+1 days forecast
        df_features[f"obsdis_{i+1}"] = df_features.obsdis.shift(-1 * (i + 1))

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

    # we keep the fist 80% for training and the last 20% for testing
    # we may want to use a random train/test split

    # here, the targets are vectors, each position corresponding to the day we are looking to predict

    split_index = int(split_ratio * df_features.shape[0])

    target_columns = ["obsdis"] + [f"obsdis_{i}" for i in range(1, DAYS_AHEAD)]
    targets = df_features[target_columns]

    f_cols = [
        f for f in df_features.columns if f.startswith("P_lag") or f.startswith("Q_lag")
    ]
    X = df_features[f_cols]

    target_columns = ["obsdis"] + [f"obsdis_{i}" for i in range(1, DAYS_AHEAD)]
    targets = df_features[target_columns]

    mo_train_x = X[0:split_index]
    mo_train_y = targets[0:split_index]

    mo_test_x = X[split_index:]
    mo_test_y = targets[split_index:]

    return mo_train_x, mo_train_y, mo_test_x, mo_test_y


def create_model(model_type, **argv):
    """
    keep it simple for now: check model name and create corresponding object
    """
    if model_type == "RandomForestRegressor":
        return MultiOutputRegressor(RandomForestRegressor(**argv))
    elif model_type == "GradientBoostingRegressor":
        return MultiOutputRegressor(GradientBoostingRegressor(**argv))
    else:
        raise ValueError("Model not implemented")


def main(data_path: str = "data/parquet.raw", model_type="RandomForestRegressor"):

    files = glob.glob(data_path + "/*.parquet")
    files = files[:10]
    scores = []
    for file in tqdm.tqdm(files):
        df = pd.read_parquet(file)
        # df = df.ffill(). this won't work in cases where most recent values are missing.
        df = df.dropna()

        df_features = create_features(df, lags=[1, 2, 3], dropna=True)

        x_train, y_train, x_test, y_test = create_train_test_splits(
            df_features, split_ratio=0.8
        )

        model = create_model(model_type=model_type, n_estimators=100, random_state=42)

        station = os.path.basename(file).split("_cleaned")[0].strip(".0")
        if x_train.shape[0] == 0:
            print(f"File for station {station} has no data")
            continue

        model.fit(x_train, y_train)

        score = model.score(x_test, y_test)
        scores.append((station, score))

    scores = pd.DataFrame(scores, columns=["station", "score"])
    scores = scores.sort_values(by="score", ascending=True)

    print(tabulate(scores.head(), headers="keys", tablefmt="fancy_outline"))
    print(tabulate(scores.tail(), headers="keys", tablefmt="fancy_outline"))

    scores.to_csv(f"scores_{model_type}.csv", index=False)


if __name__ == "__main__":

    fire.Fire(main)

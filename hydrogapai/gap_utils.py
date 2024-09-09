# HydroGAP-AI_Recursive_real_data_v3_dev - 31-08-2024
import os  # Import the os module to work with file paths

# Import core libraries
import copy
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import timedelta

# Import algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import skew, kurtosis, pearsonr
import xgboost as xgb
import lightgbm as lgb

# Import module for hyperparameter optimization
from sklearn.model_selection import GridSearchCV

# Import metrics
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import statsmodels.api as sm
import hydroeval as he

# Import Normalization modules
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import CrossCorrelation and AutoCorrelation
from statsmodels.tsa.stattools import pacf, ccf

import scipy.stats
from datetime import datetime
import os
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, r2_score
import hydroeval as he
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import copy
import random
RANDOM_STATE = 42

num_pacf_lags = 3  # Number of PACF lags to use
plag_start = 1  # how many days before should the lag period start from
num_ccf_lags = 30  # Number of CCF lags to use


np.random.RandomState().seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Calculate initial gap and streamflow metrics on the real dataset
# Extract the base file name without extension for naming outputs
def calculate_initial_metrics(data_real, input_file_path: str, results_folder: str):
    base_file_name = os.path.splitext(os.path.basename(input_file_path))[0]

    # Detect NaN values in the real data (initial dataset)
    nan_gaps = data_real["obsdis"].isna()

    # Calculate gap statistics
    missing_rate = nan_gaps.mean()
    gaps = nan_gaps.astype(int).groupby(nan_gaps.ne(nan_gaps.shift()).cumsum()).cumsum()
    total_time = (data_real.index[-1] - data_real.index[0]).days
    gap_lengths = gaps[nan_gaps == True]
    min_gap_length = gap_lengths.min() if not gap_lengths.empty else np.nan
    mean_gap_length = gap_lengths.mean() if not gap_lengths.empty else np.nan
    max_gap_length = gap_lengths.max() if not gap_lengths.empty else np.nan
    std_gap_length = gap_lengths.std() if not gap_lengths.empty else np.nan
    median_gap_length = gap_lengths.median() if not gap_lengths.empty else np.nan
    range_gap_length = (
        max_gap_length - min(gap_lengths) if not gap_lengths.empty else np.nan
    )
    nr_gaps = nan_gaps.astype(int).diff().fillna(0).abs().sum() // 2
    nr_gap_days = nan_gaps.sum()
    gap_density = nr_gap_days / total_time

    # Streamflow statistics for non-NaN values
    non_nan_values = data_real["obsdis"].dropna()
    min_value = non_nan_values.min()
    mean_value = non_nan_values.mean()
    median_value = non_nan_values.median()
    max_value = non_nan_values.max()
    std_value = non_nan_values.std()
    range_value = max_value - min_value
    skew_value = skew(non_nan_values)
    kurtosis_value = kurtosis(non_nan_values)

    # Calculate percentage of missing values per year
    data_real["year"] = data_real.index.year
    yearly_missing = data_real["obsdis"].isna().groupby(data_real["year"]).mean()

    # Remove indices where 'obsdis' is NaN for correlation calculations
    valid_indices = ~data_real["obsdis"].isna()
    tp_values = data_real.loc[valid_indices, "tp"]
    obsdis_values = data_real.loc[valid_indices, "obsdis"]
    # Calculate Pearson correlation coefficient
    pearson_corr = pearsonr(tp_values, obsdis_values)[0]

    # Plot the missing values over different periods
    data_real["missing"] = nan_gaps

    # Monthly
    fig4, ax4 = plt.subplots(figsize=(15, 6))
    data_real["missing"].resample("ME").mean().plot(ax=ax4)
    ax4.set_title(
        f"Monthly Proportion of Missing Values - {base_file_name}", fontsize=15
    )
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Proportion of Missing Values")
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig4.savefig(
        os.path.join(results_folder, f"{base_file_name}_monthly_missing_values.jpg"),
        dpi=300,
    )
    plt.close(fig4)  # Close the figure

    # Yearly
    fig5, ax5 = plt.subplots(figsize=(15, 6))
    data_real["missing"].resample("YE").mean().plot(ax=ax5)
    ax5.set_title(
        f"Yearly Proportion of Missing Values - {base_file_name}", fontsize=15
    )
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Proportion of Missing Values")
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig5.savefig(
        os.path.join(results_folder, f"{base_file_name}_yearly_missing_values.jpg"),
        dpi=300,
    )
    plt.close(fig5)  # Close the figure

    # Plot precipitation vs. time
    fig6, ax6 = plt.subplots(figsize=(15, 6))
    data_real["tp"].plot(ax=ax6, color="blue", linewidth=0.7)
    ax6.set_title(f"Total Daily Precipitation (m) - {base_file_name}", fontsize=15)
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Total Daily Precipitation (m)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig6.savefig(
        os.path.join(results_folder, f"{base_file_name}_precipitation_vs_time.jpg"),
        dpi=300,
    )
    plt.close(fig6)  # Close the figure

    # Plot streamflow vs. time with shaded regions for NaN values
    fig7, ax7 = plt.subplots(figsize=(18, 6))  # Set the figure size
    data_real["obsdis"].plot(
        ax=ax7, color="green", linewidth=1, label="Daily Streamflow (m3/d)"
    )  # Set linewidth to 1 for a thinner line
    ax7.set_title(f"Daily Streamflow (m3/d) - {base_file_name}", fontsize=15)
    ax7.set_xlabel("Time")
    ax7.set_ylabel("Daily Streamflow (m3/d)")
    plt.xticks(rotation=90)
    # Shade the NaN regions in red
    nan_groups = (
        data_real["obsdis"]
        .isna()
        .astype(int)
        .groupby(data_real["obsdis"].notna().cumsum())
        .cumsum()
    )
    nan_regions = data_real[nan_groups > 0].groupby((nan_groups == 0).cumsum())
    for _, group in nan_regions:
        ax7.axvspan(
            group.index[0], group.index[-1], color="red", alpha=0.3
        )  # Shade the NaN regions with a red color
    # Add a label to explain the NaN regions
    ax7.fill_between(
        [], [], [], color="red", alpha=0.3, label="missing streamflow values"
    )
    plt.legend()
    plt.tight_layout()
    fig7.savefig(
        os.path.join(results_folder, f"{base_file_name}_streamflow_vs_time.jpg"),
        dpi=300,
    )
    plt.close(fig7)  # Close the figure

    # Plot combined precipitation and streamflow vs. time with legend
    fig8, ax8 = plt.subplots(figsize=(18, 6))  # Set figure size
    data_real["tp"].plot(
        ax=ax8, color="blue", label="Total Daily Precipitation (m)", linewidth=1
    )  # Precipitation plot
    ax9 = ax8.twinx()  # Create a secondary axis for streamflow
    data_real["obsdis"].plot(
        ax=ax9, color="green", label="Daily Streamflow (m3/d)", linewidth=1
    )  # Streamflow plot
    ax8.set_title(
        f"Total Daily Precipitation and Streamflow vs Time - {base_file_name}",
        fontsize=15,
    )
    ax8.set_xlabel("Time")
    ax8.set_ylabel("Total Daily Precipitation (m)")
    ax9.set_ylabel("Daily Streamflow (m3/d)")
    plt.xticks(rotation=90)
    # Create legend entries for both axes
    lines1, labels1 = (
        ax8.get_legend_handles_labels()
    )  # Get labels for the precipitation axis
    lines2, labels2 = (
        ax9.get_legend_handles_labels()
    )  # Get labels for the streamflow axis
    ax8.legend(
        lines1 + lines2, labels1 + labels2, loc="upper left"
    )  # Combine both legends
    plt.tight_layout()
    fig8.savefig(
        os.path.join(
            results_folder, f"{base_file_name}_precipitation_and_streamflow_vs_time.jpg"
        ),
        dpi=300,
    )
    plt.close(fig8)  # Close the figure

    # Scatter plot of precipitation vs. streamflow
    data_clean = data_real[["tp", "obsdis"]].dropna()
    fig9, ax9 = plt.subplots(figsize=(18, 6))
    ax9.scatter(
        data_clean["tp"],
        data_clean["obsdis"],
        s=5,
        alpha=0.5,
        label=f"r={pearson_corr:.3f}",
    )
    ax9.set_title(
        f"Total Daily Precipitation vs Daily Streamflow - {base_file_name}", fontsize=15
    )
    ax9.set_xlabel("Total Daily Precipitation (m)")
    ax9.set_ylabel("Daily Streamflow (m3/d)")
    # Add trendline
    z = np.polyfit(data_clean["tp"], data_clean["obsdis"], 1)
    p = np.poly1d(z)
    ax9.plot(data_clean["tp"], p(data_clean["tp"]), "r--")
    #
    ax9.legend()
    plt.tight_layout()
    fig9.savefig(
        os.path.join(
            results_folder, f"{base_file_name}_scatter_precipitation_vs_streamflow.jpg"
        ),
        dpi=300,
    )
    plt.close(fig9)  # Close the figure

    # Return the calculated metrics
    return {
        "input_file_path": input_file_path,
        "missing_rate": missing_rate,
        "min_gap_length": min_gap_length,
        "mean_gap_length": mean_gap_length,
        "median_gap_length": median_gap_length,
        "max_gap_length": max_gap_length,
        "std_gap_length": std_gap_length,
        "range_gap_length": range_gap_length,
        "gap_density": gap_density,
        "nr_gap_days": nr_gap_days,
        "nr_gaps": nr_gaps,
        "min_value": min_value,
        "mean_value": mean_value,
        "median_value": median_value,
        "max_value": max_value,
        "std_value": std_value,
        "range_value": range_value,
        "skew_value": skew_value,
        "kurtosis_value": kurtosis_value,
    }


def introduce_synthetic_gaps(
    df, results_folder, original_filename, use_random_gaps=False
):
    try:
        # Use a different random seed just for the gap introduction process
        # if use_random_gaps:
        #    random_state = np.random.RandomState(None)  # Truly random seed
        # else:
        #    random_state = np.random.RandomState(RANDOM_STATE)  # Fixed seed for reproducibility

        #_ = np.random.RandomState(RANDOM_STATE)
        np.random.RandomState().seed(RANDOM_STATE)

        # Ensure the 'time' index is in datetime format
        df.index = pd.to_datetime(df.index)

        # Detect existing gaps and calculate gap characteristics
        gap_info_real = df["obsdis"].isna()
        data_real_gap_indices = df.index[gap_info_real]
        gap_days = gap_info_real.sum()
        max_gap = 0
        current_gap = 0
        for is_gap in gap_info_real:
            if is_gap:
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0

        total_gap_length = gap_days // 2
        min_gap_length = 1
        max_gap_length = max_gap

        # Create a copy of the DataFrame to avoid modifying the original data
        df_copy = copy.deepcopy(df)
        remaining_gap_length = total_gap_length
        used_dates = set(df_copy.index[gap_info_real])

        while remaining_gap_length > 0:
            # Randomly select a gap length within the specified range
            gap_length = random.randint(
                min_gap_length, min(max_gap_length, remaining_gap_length)
            )

            # Try to find a non-overlapping start date
            max_attempts = 100
            attempts = 0
            while attempts < max_attempts:
                start_date = random.choice(df_copy.index)
                end_date = start_date + timedelta(days=gap_length - 1)

                # Ensure the end date does not exceed the dataset's date range
                if end_date > df_copy.index[-1]:
                    end_date = df_copy.index[-1]
                    gap_length = (end_date - start_date).days + 1

                # Check for overlap with existing gaps
                overlapping = any(
                    d in used_dates
                    for d in pd.date_range(start=start_date, end=end_date)
                )
                if not overlapping:
                    break
                attempts += 1

            if attempts >= max_attempts:
                print(
                    f"Warning: Could not find a non-overlapping gap after {max_attempts} attempts."
                )
                break

            # Mark the gap dates as used
            used_dates.update(pd.date_range(start=start_date, end=end_date))

            # Introduce the synthetic gap by setting 'obsdis' values to NaN
            df_copy.loc[start_date:end_date, "obsdis"] = np.nan

            # Subtract the gap length from the remaining gap length
            remaining_gap_length -= gap_length

        # Generate the output filename using the original dataset's filename
        base_filename = os.path.splitext(original_filename)[0]  # Remove the extension
        output_csv_path = os.path.join(
            results_folder,
            f"{base_filename}_gapped_{total_gap_length}d_min{min_gap_length}d_max{max_gap_length}d.csv",
        )

        # Save the modified DataFrame to a CSV file with appropriate suffix
        df_copy.to_csv(output_csv_path, na_rep="NaN", encoding="utf-8")

        return df_copy, data_real_gap_indices

    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return None


def detect_gaps(data, column="obsdis"):
    # Detect gaps
    nan_mask = data[column].isna()
    # Identify unique gap segments by giving each gap a unique identifier
    gap_segment_ids = (nan_mask != nan_mask.shift()).cumsum() * nan_mask
    data["gap_id"] = gap_segment_ids
    # Filter out zero values (non-gap periods) and get unique gap IDs
    unique_gap_ids = data["gap_id"].loc[data["gap_id"] != 0].unique()
    gap_info = []
    days_to_gaps = []
    for gap_id in unique_gap_ids:
        gap_data = data[data["gap_id"] == gap_id]
        gap_start = gap_data.index[0]
        gap_end = gap_data.index[-1]
        gap_duration = len(gap_data)  # Duration of the gap in days
        gap_info.append(
            {
                "gap_id": gap_id,
                "start_date": gap_start,
                "duration_days": gap_duration,
                "end_date": gap_end,
            }
        )
        # Calculate the number of days from the start of the time series to the start of the current gap
        days_to_gap = (gap_start - data.index.min()).days + 1
        days_to_gaps.append(days_to_gap)

    return gap_info, days_to_gaps


def getTopCorrelations(df, col_Dis, col_Prec, n_pacf, n_ccf):
    # Calculate PACF values for autocorrelation
    pacf_values = pacf(df[col_Dis], nlags=n_pacf)
    pacf_values = pd.Series(
        pacf_values[1:], index=np.arange(1, len(pacf_values))
    )  # Exclude lag 0

    # Filter PACF values with correlation coefficient > 0.5
    filtered_pacf = pacf_values[abs(pacf_values) > 0.5]

    # Select up to n_pacf lags
    top_pacf_lags = filtered_pacf.head(n_pacf).index.tolist()

    # Calculate CCF values for cross-correlation
    ccf_vals = ccf(df[col_Dis], df[col_Prec])
    ccf_vals = pd.Series(
        ccf_vals[1:], index=np.arange(1, len(ccf_vals))
    )  # Exclude lag 0

    # Filter CCF values with correlation coefficient > 0.5
    filtered_ccf = ccf_vals[abs(ccf_vals) > 0.5]

    # Select up to n_ccf lags
    top_ccf_lags = filtered_ccf.head(n_ccf).index.tolist()

    return top_pacf_lags, top_ccf_lags, pacf_values, ccf_vals


def preprocess_data(df, start_year, end_year, Q_lags=[], P_lags=[]):
    def normalization(df):
        scaler_tp = StandardScaler()
        scaler_obsdis = StandardScaler()

        scaled_tp = scaler_tp.fit_transform(df[["tp"]])
        scaled_obsdis = scaler_obsdis.fit_transform(df[["obsdis"]])

        scaled_df = pd.DataFrame(
            {"tp": scaled_tp.flatten(), "obsdis": scaled_obsdis.flatten()},
            index=df.index,
        )

        return scaled_df, scaler_tp, scaler_obsdis

    def create_features(df):
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["year"] = df.index.year
        return df

    def create_Q_lag_features(df, lags):
        for lag in lags:
            df[f"Q_lag_time_{lag}"] = df["obsdis"].shift(periods=lag)
        return df

    def create_P_lag_features(df, lags):
        for lag in lags:
            df[f"P_lag_time_{lag}"] = df["tp"].shift(periods=lag)
        return df

    def create_dummy_variables(df):
        # Include all possible months, quarters, and years you expect in the data
        df["year"] = df["year"].astype(
            pd.CategoricalDtype(categories=range(start_year, end_year))
        )  # Adjust years as needed
        df = pd.get_dummies(df, columns=["month", "quarter", "year"], dtype=float)
        return df

    df, scaler_tp, scaler_obsdis = normalization(df)
    df = create_features(df)

    if Q_lags:
        df = create_Q_lag_features(df, Q_lags)

    if P_lags:
        df = create_P_lag_features(df, P_lags)

    df = create_dummy_variables(df)

    return df, scaler_tp, scaler_obsdis


def train_model(x_train, y_train, model_type):
    # Train a model based on the specified type.
    if model_type == "rf":
        model = RandomForestRegressor(random_state=RANDOM_STATE)
    elif model_type == "xgb":
        model = xgb.XGBRegressor(
            objective="reg:squarederror", eval_metric="rmse", random_state=RANDOM_STATE
        )
    elif model_type == "lgb":
        model = lgb.LGBMRegressor(
            objective="regression", metric="rmse", random_state=RANDOM_STATE
        )
    elif model_type == "lr":
        model = SGDRegressor(random_state=RANDOM_STATE)
    elif model_type == "knn":
        model = KNeighborsRegressor()
    elif model_type == "svr":
        model = SVR()
    else:
        raise ValueError(
            "Unsupported model type. Choose from 'rf', 'xgb', 'lgb', 'lr', 'knn', 'svr'."
        )

    model.fit(x_train.values, y_train.values)
    return model


def train_model_hyper_opt(x_train, y_train, model_type):
    # Train a model based on the specified type.
    if model_type == "rf":
        model = RandomForestRegressor(random_state=RANDOM_STATE)
        param_grid = {
            "n_estimators": [50, 100, 200, 300, 400, 500],
            "max_depth": [2, 4, 8, 10, 15],
            "min_samples_split": [1, 2, 4, 8],
            "min_samples_leaf": [1, 2, 4, 8],
        }
    elif model_type == "xgb":
        model = xgb.XGBRegressor(
            objective="reg:squarederror", eval_metric="rmse", random_state=RANDOM_STATE
        )
        param_grid = {
            "n_estimators": [50, 100, 200, 300, 400, 500],
            "max_depth": [2, 4, 8, 10, 15],
            "learning_rate": [0.001, 0.01, 0.1, 0.5, 1],
            "subsample": [0.2, 0.4, 0.6, 0.8, 1],
        }
    elif model_type == "lgb":
        model = lgb.LGBMRegressor(
            objective="regression", metric="rmse", random_state=RANDOM_STATE
        )
        param_grid = {
            "n_estimators": [50, 100, 200, 300, 400, 500],
            "max_depth": [2, 4, 8, 10, 15],
            "learning_rate": [0.001, 0.01, 0.1, 0.5, 1],
            "num_leaves": [5, 10, 20, 40, 80, 160],
        }
    elif model_type == "lr":
        model = SGDRegressor(random_state=RANDOM_STATE, shuffle=False)
        param_grid = {
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "penalty": ["l2", "l1", "elasticnet"],
            "max_iter": [100, 500, 1000, 2000],
        }
    elif model_type == "knn":
        model = KNeighborsRegressor()
        param_grid = {
            "n_neighbors": [3, 5, 10, 20],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        }
    elif model_type == "svr":
        model = SVR()
        param_grid = {
            "C": [0.01, 0.1, 0.2, 0.4, 0.8, 1],
            "kernel": [ "rbf", "sigmoid"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            "epsilon": [0.01, 0.1, 0.2, 0.5, 1],
        }

    else:
        raise ValueError(
            "Unsupported model type. Choose from 'rf', 'xgb', 'lgb', 'lr', 'knn', 'svr'."
        )

    grid_search = GridSearchCV(
        model, param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=1
    )
    grid_search.fit(x_train.values, y_train.values)

    model = grid_search.best_estimator_
    model.fit(x_train.values, y_train.values)

    return model


def update_features(previous_input, new_prediction, num_lags):
    updated_input = previous_input.copy()
    # Update Q lag features dynamically based on the number of lags
    for i in range(num_lags - 1, 0, -1):
        updated_input[i] = updated_input[i - 1]  # Shift lag values
    updated_input[0] = new_prediction  # Set the new prediction as the first lag
    return updated_input


# METRICS
def index_of_agreement(obs, cal):
    mean_obs = np.mean(obs)
    ioa = 1 - (np.sum((obs - cal) ** 2)) / (
        np.sum((np.abs(cal - mean_obs) + np.abs(obs - mean_obs)) ** 2)
    )
    return ioa


def recursive_forecasting(initial_input, model, n_steps, gaps_df):
    predictions = []
    current_input = copy.deepcopy(initial_input.flatten())

    for step in range(n_steps):
        # Predict the next value
        next_prediction = model.predict(current_input.reshape(1, -1))[0]
        predictions.append(next_prediction)

        # Update the input features for the next step
        if step + 1 < len(gaps_df):
            # Update Q lag values
            updated_q_lags = update_features(
                current_input[:num_pacf_lags], next_prediction, num_pacf_lags
            )

            # Retain P lag values from the corresponding row in gaps_df
            p_lags = gaps_df.iloc[step + 1, num_pacf_lags:].values

            # Concatenate updated Q lags and unchanged P lags
            current_input = np.concatenate((updated_q_lags, p_lags))
        else:
            # If we are at the last step, keep updating with the predicted values
            updated_q_lags = update_features(
                current_input[:num_pacf_lags], next_prediction, num_pacf_lags
            )
            current_input[:num_pacf_lags] = updated_q_lags

    return predictions


import scipy.stats


def process_file(
    input_file_path: str,
    results_folder: str,
    model_type: str = "rf",
    hyper_opt: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.Series]:
    # Initialize/reset variables specific to each file

    filename = os.path.basename(input_file_path)
    # Read data for predictions.
    data_real = pd.read_csv(input_file_path, parse_dates=["time"], index_col="time")

    # data_real = pd.read_parquet(input_file_path)

    # Read data for gap metrics

    # Call the initial metrics calculation function
    metrics_gaps = calculate_initial_metrics(data_real, input_file_path, results_folder)

    # Introduce synthetic gaps
    data, data_real_gap_indices = introduce_synthetic_gaps(
        data_real, results_folder, filename
    )

    # Automatically detect start_year and end_year
    start_year = int(data.index.year.min())
    end_year = int(data.index.year.max())

    gap_info, days_to_gaps = detect_gaps(data)
    if not days_to_gaps:
        raise ValueError("Station file has no gaps")

    # If there is one or more gaps in streamflow data early on inside the P_lags or Q_lags range, the recursive method cannot be initiated,
    # so the spline interpolation method is used to imputate the early gaps
    if max(num_ccf_lags, num_pacf_lags) > days_to_gaps[0]:
        # Determine the range for interpolation
        interpolation_range = max(num_ccf_lags, days_to_gaps[0])
        # Perform spline interpolation to fill the gaps in the first interpolation_range values
        data["obsdis"].iloc[:interpolation_range] = (
            data["obsdis"]
            .iloc[:interpolation_range]
            .interpolate(method="spline", order=3)
        )
        # If spline interpolation does not work, perform linear interpolation
        data["obsdis"].iloc[:interpolation_range] = (
            data["obsdis"].iloc[:interpolation_range].interpolate(method="linear")
        )
        # Backfill remaining NaNs if interpolation does not cover all gap
        data["obsdis"].iloc[:interpolation_range].fillna(method="bfill", inplace=True)

        data_real_gap_indices = data_real_gap_indices.drop(
            data_real_gap_indices[: interpolation_range - days_to_gaps[0] + 1]
        )

 
    gap_info, days_to_gaps = detect_gaps(data)

    # Convert gap_info to DataFrame for easy manipulation
    gap_info_df = pd.DataFrame(gap_info)

    # Preprocess the data
    data = data[["tp", "obsdis"]]
    data_without_NaN = data.dropna()

    _, _, pacf_vals, ccf_vals = getTopCorrelations(
        data_without_NaN, "obsdis", "tp", num_pacf_lags, num_ccf_lags
    )

    Q_lags = list(range(1, num_pacf_lags + 1))
    P_lags = list(range(plag_start, num_ccf_lags + 1))

    training, scaler_tp, scaler_obsdis = preprocess_data(
        data_without_NaN, start_year, end_year, Q_lags=Q_lags, P_lags=P_lags
    )
    training = training.dropna()

    ## TODO: use scaler_tp when using SVC-like models
    data_with_gaps, scaler_tp, scaler_obsdis = preprocess_data(
        data, start_year, end_year, Q_lags=Q_lags, P_lags=P_lags
    )

    # Training
    steps = 1
    data_train = training[:-steps]

    # TODO: re-evaluate testing?
    # data_test = training[-steps:]

    x_train = data_train.drop(columns=["obsdis", "tp"])
    y_train = data_train["obsdis"]

    if hyper_opt == True:
        model = train_model_hyper_opt(x_train, y_train, model_type=model_type)
    else:
        model = train_model(x_train, y_train, model_type=model_type)

    combined_dfs = pd.DataFrame()
    for _, gap in gap_info_df.iterrows():
        gap_start = gap["start_date"]

        gap_end = gap["end_date"]

        # Extract the features for the current gap
        gap_features = data_with_gaps.loc[gap_start:gap_end]
        gap_features = gap_features.drop(["tp", "obsdis"], axis=1)
        # Get the initial input for the recursive forecasting
        initial_test_input = gap_features.iloc[0].values.reshape(1, -1)
        # Number of steps ahead to predict
        n_steps_ahead = len(gap_features)
        # Perform recursive forecasting for the current gap
        multi_step_predictions = recursive_forecasting(
            initial_test_input, model, n_steps_ahead, gap_features
        )

        # Assign the predictions back to the original data
        data.loc[gap_start:gap_end, "obsdis"] = multi_step_predictions
        data_with_gaps.loc[gap_start:gap_end, "obsdis"] = multi_step_predictions

        # Generalize the update of Q_lag_time columns for any number of lags
        for lag in range(1, num_pacf_lags + 1):
            lag_column_name = f"Q_lag_time_{lag}"
            indices = pd.date_range(
                start=gap_start + timedelta(days=lag),
                periods=len(multi_step_predictions),
            )
            # Check if all the indices exist in data_with_gaps
            valid_indices = indices.intersection(data_with_gaps.index)
            if not valid_indices.empty:
                data_with_gaps.loc[valid_indices, lag_column_name] = (
                    multi_step_predictions[: len(valid_indices)]
                )

        # Denormalize the predictions
        multi_step_predictions_denorm = scaler_obsdis.inverse_transform(
            np.array(multi_step_predictions).reshape(-1, 1)
        )
        # Apply non-negative correction after denormalization
        multi_step_predictions_denorm = np.maximum(0, multi_step_predictions_denorm)
        # Convert to DataFrame
        multi_step_predictions_denorm = pd.DataFrame(
            multi_step_predictions_denorm,
            columns=["predicted"],
            index=pd.date_range(
                start=gap_start, periods=len(multi_step_predictions_denorm)
            ),
        )

        # Load the original, ungapped data for comparison
        val = data_real["obsdis"].loc[gap_start:gap_end]
        val = pd.DataFrame(val)
        dates = pd.date_range(start=gap_start, periods=len(val), freq="D")
        val = pd.DataFrame(val, index=dates)
        # Combine the denormalized predictions with the actual values
        combined_df = pd.concat([val, multi_step_predictions_denorm], axis=1)
        # Append the combined_df to the list
        combined_dfs = pd.concat(
            [combined_dfs, combined_df]
        )  # col1=real values+gaps for indexes of all gaps; col2=all preds

    all_combined_dfs = combined_dfs.drop(index=data_real_gap_indices)

    val_full = data_real[["obsdis"]].copy()  # Create a copy of the 'obsdis' column
    val_full["predicted"] = val_full[
        "obsdis"
    ]  # Duplicate the 'obsdis' column and rename it 'predicted'
    val_full.update(
        combined_dfs["predicted"]
    )  # Update the 'predicted' column in val_full with values from combined_dfs where dates match

    # real_predictions.loc[data_real_gap_indices, 'obsdis'] = val_full.loc[data_real_gap_indices, 'predicted']
    real_predictions = val_full.loc[data_real_gap_indices, "predicted"]

    if not all_combined_dfs.empty:
        # Calculate MAE, MBE, and percentage error
        mae = mean_absolute_error(
            all_combined_dfs["obsdis"], all_combined_dfs["predicted"]
        )
        mbe = np.mean(all_combined_dfs["predicted"] - all_combined_dfs["obsdis"])
        percentage_error = (
            np.mean(
                np.abs(all_combined_dfs["obsdis"] - all_combined_dfs["predicted"])
                / all_combined_dfs["obsdis"]
            )
            * 100
        )

        # Other metrics only if enough data points exist
        if len(all_combined_dfs["obsdis"]) >= 2:
            r2 = r2_score(all_combined_dfs["obsdis"], all_combined_dfs["predicted"])
            rmse = np.sqrt(
                np.mean(
                    (all_combined_dfs["predicted"] - all_combined_dfs["obsdis"]) ** 2
                )
            )
            nash_sutcliffe = float(
                he.evaluator(
                    he.nse, all_combined_dfs["predicted"], all_combined_dfs["obsdis"]
                )[0]
            )
            ioa = index_of_agreement(
                all_combined_dfs["obsdis"], all_combined_dfs["predicted"]
            )
            pearson_corr = scipy.stats.pearsonr(
                all_combined_dfs["obsdis"], all_combined_dfs["predicted"]
            )[0]
        else:
            # If not enough data points, set them to None
            r2 = rmse = nash_sutcliffe = ioa = pearson_corr = None
    else:
        # If the dataframe is empty, set all metrics to None
        mae = mbe = percentage_error = r2 = rmse = nash_sutcliffe = ioa = (
            pearson_corr
        ) = None

    # KGE values are computed only if data is available
    if len(all_combined_dfs) > 0:
        kge_values = he.evaluator(
            he.kge, all_combined_dfs["predicted"], all_combined_dfs["obsdis"]
        )
        kge_overall = float(kge_values[0])
        kge_correlation = float(kge_values[1])
        kge_bias = float(kge_values[2])
        kge_variability = float(kge_values[3])
    else:
        kge_overall = kge_correlation = kge_bias = kge_variability = None

    # Now build the metrics dictionary
    ml_metrics_gaps = {
        "Q_lags": Q_lags,  # Add Q_lags array to metrics
        "Q_lags_Coefficients": list([
            round(pacf_vals[lag], 3) for lag in Q_lags
        ]),  # Add Q_lags correlation coefficients rounded to 3 decimals
        "P_lags": P_lags,  # Add P_lags array to metrics
        "P_lags_Coefficients": [
            round(ccf_vals[lag], 3) for lag in P_lags
        ],  # Add P_lags correlation coefficients rounded to 3 decimals
        "R2_score": r2,
        "RMSE": rmse,
        "Mean Bias Error": mbe,  # MBE calculated regardless of gap count
        "MAE": mae,  # MAE calculated regardless of gap count
        "Percentage Error (%)": percentage_error,  # Added Percentage Error
        "Nash-Sutcliffe": nash_sutcliffe,
        "Index of Agreement": ioa,
        "Correlation Coefficient": pearson_corr,
        "KGE Overall": kge_overall,
        "KGE Correlation": kge_correlation,
        "KGE Bias": kge_bias,
        "KGE Variability": kge_variability,
    }

    # Add gap metrics to your final metrics or export them as CSVs
    metrics_gaps.update(ml_metrics_gaps)
    return all_combined_dfs, val_full, metrics_gaps, real_predictions
